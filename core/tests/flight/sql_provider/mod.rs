use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt};
use arrow_flight::FlightDescriptor;
use async_trait::async_trait;
use datafusion::arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::physical_plan::{collect, ExecutionPlan, ExecutionPlanProperties};
use datafusion::prelude::{col, lit, SessionConfig, SessionContext};
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_table_providers::flight::codec::FlightPhysicalCodec;
use datafusion_table_providers::flight::sql::{FlightSqlDriver, PASSWORD, QUERY, USERNAME};
use datafusion_table_providers::flight::{
    FlightChannelProvider, FlightFactoryOptions, FlightProperties, FlightSqlSource,
    FlightSqlTableFactory, FlightSqlTableOptions, SizeLimits,
};
use futures::{StreamExt, TryStreamExt};
use prost::Message;
use rstest::rstest;
use tonic::transport::Channel;
use tonic::Status;

mod fixture;

use fixture::{MetadataResponse, RecordedRequest, RequestGate, SqlProviderFixture, Submission};

fn factory() -> FlightSqlTableFactory {
    FlightSqlTableFactory::new(FlightSqlDriver::new().with_persistent_headers(true)).with_options(
        FlightFactoryOptions::new()
            .with_properties(FlightProperties::new().with_reusable_flight_info(true)),
    )
}

fn options() -> HashMap<String, String> {
    HashMap::from([
        (
            "flight.sql.header.authorization".into(),
            "Bearer fixture-token".into(),
        ),
        ("flight.sql.header.x-fixture-scope".into(), "test".into()),
    ])
}

fn int_values(batches: &[RecordBatch], column: usize) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}

async fn optimizer_fixture(
    source: FlightSqlSource,
) -> Result<(SqlProviderFixture, SessionContext, SessionContext)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, true),
        Field::new("enabled", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10])),
            Arc::new(Int64Array::from(vec![
                None,
                Some(-10),
                Some(0),
                Some(10),
                Some(20),
                Some(20),
                Some(i64::MAX),
                Some(i64::MIN),
                Some(-1),
                Some(1),
                None,
            ])),
            Arc::new(BooleanArray::from(vec![
                None,
                Some(false),
                Some(true),
                None,
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                None,
                Some(true),
                Some(true),
            ])),
        ],
    )?;
    // Put identical rows in different batches so comparisons must retain multiplicity.
    let table = Arc::new(MemTable::try_new(
        schema.clone(),
        vec![vec![batch.slice(0, 5), batch.slice(5, 6)]],
    )?);
    let empty = Arc::new(MemTable::try_new(schema, vec![vec![]])?);
    // Avoid reordering unordered LIMIT/OFFSET inputs without adding a sort barrier.
    let config = SessionConfig::new().with_target_partitions(1);
    let server = SessionContext::new_with_config(config.clone());
    let baseline = SessionContext::new_with_config(config.clone());
    for context in [&server, &baseline] {
        context.register_table("cases", table.clone())?;
        context.register_table("empty_cases", empty.clone())?;
    }
    let local_source = match &source {
        FlightSqlSource::Table(name) => baseline.table(name.as_str()).await?,
        FlightSqlSource::Query(query) => baseline.sql(query).await?,
    };
    baseline.register_table("remote", local_source.into_view())?;
    let fixture = SqlProviderFixture::start_with_context(server).await?;
    let remote = factory()
        .with_table_options(FlightSqlTableOptions::default().with_datafusion_compatible_pushdown())
        .open_table(fixture.endpoint(), source, options())
        .await?;
    let optimized = SessionContext::new_with_config(config);
    optimized.register_table("remote", Arc::new(remote))?;
    Ok((fixture, optimized, baseline))
}

async fn sorted_rows(context: &SessionContext, sql: &str) -> Result<Vec<Vec<ScalarValue>>> {
    let batches = context.sql(sql).await?.collect().await?;
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|column| ScalarValue::try_from_array(column, row))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    rows.sort_by_cached_key(|row| format!("{row:?}"));
    Ok(rows)
}

async fn assert_query_matches_baseline(
    fixture: &SqlProviderFixture,
    optimized: &SessionContext,
    baseline: &SessionContext,
    sql: &str,
) -> Result<String> {
    let start = fixture.requests().len();
    let expected = sorted_rows(baseline, sql).await?;
    let actual = sorted_rows(optimized, sql).await?;
    let requests = fixture.requests();
    let scans = requests[start..]
        .iter()
        .filter_map(|request| request.sql.as_deref())
        .collect::<Vec<_>>();
    assert_eq!(scans.len(), 1, "{sql}: {requests:?}");
    assert_eq!(actual, expected, "{sql}\nscan: {}", scans[0]);
    Ok(scans[0].to_owned())
}

#[tokio::test]
async fn residual_filters_prevent_remote_limits() -> Result<()> {
    for source in [
        FlightSqlSource::table("cases")?,
        FlightSqlSource::query("SELECT * FROM cases")?,
    ] {
        let table_source = matches!(source, FlightSqlSource::Table(_));
        let (fixture, optimized, baseline) = optimizer_fixture(source).await?;
        for (predicate, pushable) in [
            ("score >= 0", true),
            ("length(CAST(score AS VARCHAR)) = 2", false),
            ("score >= 0 AND length(CAST(score AS VARCHAR)) = 2", true),
        ] {
            for bound in ["LIMIT 1", "LIMIT 2 OFFSET 1"] {
                let sql = format!("SELECT id FROM remote WHERE {predicate} {bound}");
                let scan =
                    assert_query_matches_baseline(&fixture, &optimized, &baseline, &sql).await?;
                assert!(!scan.contains(" LIMIT "), "{scan}");
                assert!(!scan.contains(" OFFSET "), "{scan}");
                assert_eq!(scan.contains(" WHERE "), table_source && pushable, "{scan}");
                assert!(!scan.contains("length("), "{scan}");
            }
        }
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn inexact_filters_keep_residual_columns() -> Result<()> {
    let (fixture, optimized, baseline) =
        optimizer_fixture(FlightSqlSource::table("cases")?).await?;
    let scan = assert_query_matches_baseline(
        &fixture,
        &optimized,
        &baseline,
        "SELECT r.id FROM remote AS r WHERE r.score >= 10 LIMIT 2",
    )
    .await?;
    assert!(
        scan.starts_with("SELECT \"id\", \"score\" FROM \"cases\" WHERE "),
        "{scan}"
    );
    assert!(!scan.contains("\"enabled\""), "{scan}");
    assert!(!scan.contains("\"r\"."), "{scan}");
    assert!(!scan.contains(" LIMIT "), "{scan}");
    fixture.shutdown().await
}

#[tokio::test]
async fn boolean_trees_have_no_false_negatives() -> Result<()> {
    for source in [
        FlightSqlSource::table("cases")?,
        FlightSqlSource::query("SELECT * FROM cases")?,
    ] {
        let table_source = matches!(source, FlightSqlSource::Table(_));
        let (fixture, optimized, baseline) = optimizer_fixture(source).await?;
        for (predicate, pushable) in [
            ("score IS NULL OR score = -9223372036854775808", true),
            ("score IN (-9223372036854775808, 9223372036854775807)", true),
            ("NOT (score > 0 OR enabled)", true),
            ("(score < 0 OR enabled) AND NOT (score = 0)", true),
            ("enabled IS TRUE OR enabled IS NULL", true),
            ("NOT enabled", true),
            ("(score >= 0 AND enabled) OR score IS NULL", true),
            ("score < 0 OR score >= 10 AND enabled = false", true),
            ("score = 20 OR length(CAST(id AS VARCHAR)) = 2", false),
            ("NOT (score > 0 AND length(CAST(id AS VARCHAR)) = 1)", false),
        ] {
            for bound in ["", "LIMIT 1", "LIMIT 2 OFFSET 1"] {
                let sql =
                    format!("SELECT id, score, enabled FROM remote WHERE {predicate} {bound}");
                let scan =
                    assert_query_matches_baseline(&fixture, &optimized, &baseline, &sql).await?;
                assert_eq!(scan.contains(" WHERE "), table_source && pushable, "{scan}");
                assert!(!scan.contains(" LIMIT "), "{scan}");
                assert!(!scan.contains(" OFFSET "), "{scan}");
            }
        }
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn query_outer_filter_is_never_remote() -> Result<()> {
    let query = "SELECT id, score, enabled FROM cases WHERE score IS NOT NULL ORDER BY id LIMIT 6";
    let (fixture, optimized, baseline) = optimizer_fixture(FlightSqlSource::query(query)?).await?;
    for bound in ["LIMIT 1", "LIMIT 2 OFFSET 1"] {
        let sql = format!("SELECT id FROM remote WHERE score >= 20 {bound}");
        let scan = assert_query_matches_baseline(&fixture, &optimized, &baseline, &sql).await?;
        assert_eq!(scan, format!("SELECT \"id\", \"score\" FROM ({query})"));
    }
    fixture.shutdown().await
}

#[tokio::test]
async fn constant_and_count_outputs_preserve_source_multiplicity() -> Result<()> {
    for source in [
        FlightSqlSource::table("cases")?,
        FlightSqlSource::table("empty_cases")?,
        FlightSqlSource::query("SELECT * FROM cases")?,
        FlightSqlSource::query("SELECT * FROM empty_cases")?,
    ] {
        let (fixture, optimized, baseline) = optimizer_fixture(source).await?;
        for sql in [
            "SELECT COUNT(*) AS n FROM remote",
            "SELECT 1 AS one FROM remote",
        ] {
            let scan = assert_query_matches_baseline(&fixture, &optimized, &baseline, sql).await?;
            assert!(scan.starts_with("SELECT 1 FROM "), "{scan}");
            assert!(!scan.contains(" WHERE "), "{scan}");
            assert!(!scan.contains(" LIMIT "), "{scan}");
            assert!(!scan.contains(" OFFSET "), "{scan}");
        }
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn opted_in_predicates_are_inexact_and_rendered_for_table_sources_only() -> Result<()> {
    for (source, enabled, pushed) in [
        (FlightSqlSource::table("t")?, true, true),
        (FlightSqlSource::table("t")?, false, false),
        (FlightSqlSource::query("SELECT * FROM t")?, true, false),
    ] {
        let fixture = SqlProviderFixture::start().await?;
        let table_options = if enabled {
            FlightSqlTableOptions::default().with_datafusion_compatible_pushdown()
        } else {
            FlightSqlTableOptions::default()
        };
        let table = factory()
            .with_table_options(table_options)
            .open_table(fixture.endpoint(), source, options())
            .await?;
        let filter = col("a").gt(lit(1_i64));
        assert_eq!(
            table.supports_filters_pushdown(&[&filter])?,
            vec![if pushed {
                TableProviderFilterPushDown::Inexact
            } else {
                TableProviderFilterPushDown::Unsupported
            }]
        );
        assert_eq!(fixture.requests().len(), 1);
        let ctx = SessionContext::new();
        ctx.register_table("remote", Arc::new(table))?;
        let batches = ctx
            .sql("SELECT r.a FROM remote AS r WHERE (r.a > 1 AND r.c < 45) OR r.c IS NULL LIMIT 1")
            .await?
            .collect()
            .await?;
        assert_eq!(int_values(&batches, 0), vec![Some(2)]);
        let requests = fixture.requests();
        let scan = requests
            .iter()
            .filter_map(|r| r.sql.as_deref())
            .nth(1)
            .unwrap();
        assert_eq!(scan.contains(" WHERE "), pushed, "{scan}");
        assert!(!scan.contains(" LIMIT "), "{scan}");
        assert!(!scan.contains("\"r\"."), "{scan}");
        if pushed {
            assert!(
                scan.starts_with("SELECT \"a\", \"c\" FROM \"t\" WHERE "),
                "{scan}"
            );
            assert!(!scan.contains("flight_source"), "{scan}");
        }
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn renderable_predicates_and_allowlisted_functions_are_pushed() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .with_table_options(FlightSqlTableOptions::default().with_datafusion_compatible_pushdown())
        .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
        .await?;
    let ctx = SessionContext::new();
    ctx.register_table("remote", Arc::new(table))?;
    for (predicate, expected, pushed) in [
        ("a > c", vec![], true),
        ("abs(c - 35) < 10", vec![Some(3), Some(4)], true),
        ("lower(CAST(a AS VARCHAR)) = '2'", vec![Some(2)], true),
        (
            "CAST(c AS DOUBLE) > 25.5",
            vec![Some(3), Some(4), Some(5)],
            true,
        ),
        ("coalesce(c, 0) = 0", vec![Some(2)], true),
        (
            "a = 2 OR length(CAST(a AS VARCHAR)) > 0",
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
            false,
        ),
    ] {
        let start = fixture.requests().len();
        let batches = ctx
            .sql(&format!(
                "SELECT a FROM remote WHERE {predicate} ORDER BY a"
            ))
            .await?
            .collect()
            .await?;
        assert_eq!(int_values(&batches, 0), expected, "{predicate}");
        let requests = fixture.requests();
        let scan = requests[start..]
            .iter()
            .find_map(|r| r.sql.as_deref())
            .unwrap();
        assert_eq!(scan.contains(" WHERE "), pushed, "{scan}");
    }
    fixture.shutdown().await
}

#[tokio::test]
async fn pushed_predicates_preserve_literal_names_and_strings() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .with_table_options(FlightSqlTableOptions::default().with_datafusion_compatible_pushdown())
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("literal_names")?,
            options(),
        )
        .await?;
    let ctx = SessionContext::new();
    ctx.register_table("remote", Arc::new(table))?;
    let batches = ctx
        .sql("SELECT \"a\"\"\"\"b\" FROM remote WHERE \"a\"\"\"\"b\" = 'a''''b'")
        .await?
        .collect()
        .await?;
    let values = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec![Some("a''b")]);
    assert_eq!(
        fixture
            .requests()
            .iter()
            .filter_map(|r| r.sql.as_deref())
            .nth(1),
        Some("SELECT \"a\"\"\"\"b\" FROM \"literal_names\" WHERE (\"a\"\"\"\"b\" = 'a''''b')")
    );
    fixture.shutdown().await
}

#[tokio::test]
async fn discovery_is_full_schema_and_never_fetched() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table(" t ")?,
            options(),
        )
        .await?;
    assert_eq!(fixture.requests().len(), 1);
    assert_eq!(fixture.requests()[0].operation, "GetFlightInfo");
    assert_eq!(
        fixture.requests()[0].sql.as_deref(),
        Some("SELECT * FROM \"t\" LIMIT 1")
    );
    assert_eq!(
        table
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
    assert_eq!(table.table_type(), TableType::View);
    assert!(table.statistics().is_none());
    let filters = [col("a").gt(lit(2_i64)), col("b").is_null()];
    assert_eq!(
        table.supports_filters_pushdown(&filters.iter().collect::<Vec<_>>())?,
        vec![TableProviderFilterPushDown::Unsupported; 2]
    );
    let ctx = SessionContext::new();
    assert!(table
        .scan(&ctx.state(), Some(&vec![3]), &[], None)
        .await
        .is_err());
    assert_eq!(fixture.requests().len(), 1);
    fixture.shutdown().await
}

#[tokio::test]
async fn scans_get_fresh_projected_tickets() -> Result<()> {
    for (source, base) in [
        (FlightSqlSource::table("t")?, "\"t\""),
        (
            FlightSqlSource::query("SELECT a, b, c FROM t;")?,
            "(SELECT a, b, c FROM t)",
        ),
    ] {
        let fixture = SqlProviderFixture::start().await?;
        let table = factory()
            .open_table(fixture.endpoint(), source, options())
            .await?;
        let ctx = SessionContext::new();
        ctx.register_table("remote", Arc::new(table))?;
        for (query, selection, names, expected) in [
            (
                "SELECT c, a FROM remote",
                "\"a\", \"c\"",
                vec!["c", "a"],
                vec![
                    vec![Some(10), None, Some(30), Some(40), Some(50)],
                    vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
                ],
            ),
            (
                "SELECT a AS x, a AS y FROM remote",
                "\"a\"",
                vec!["x", "y"],
                vec![vec![Some(1), Some(2), Some(3), Some(4), Some(5)]; 2],
            ),
            (
                "SELECT 1 AS one FROM remote",
                "1",
                vec!["one"],
                vec![vec![Some(1); 5]],
            ),
        ] {
            let plan = ctx.sql(query).await?.create_physical_plan().await?;
            assert_eq!(
                plan.schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect::<Vec<_>>(),
                names
            );
            let last = fixture.requests().last().unwrap().clone();
            assert_eq!(last.operation, "GetFlightInfo");
            assert_eq!(last.sql, Some(format!("SELECT {selection} FROM {base}")));
            let batches = collect(plan, ctx.task_ctx()).await?;
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
            for batch in &batches {
                assert_eq!(batch.num_columns(), names.len());
            }
            for (column, values) in expected.into_iter().enumerate() {
                assert_eq!(int_values(&batches, column), values);
            }
            let request = fixture.requests().last().unwrap().clone();
            assert_eq!(request.operation, "DoGet");
            assert_eq!(request.sql, None);
            assert_eq!(request.ticket_ids, last.ticket_ids);
        }
        let requests = fixture.requests();
        let submissions = requests
            .iter()
            .filter(|r| r.operation == "GetFlightInfo")
            .collect::<Vec<_>>();
        assert_eq!(submissions.len(), 4);
        let tickets = submissions
            .iter()
            .map(|r| r.ticket_ids[0])
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(tickets.len(), 4);
        assert_eq!(
            requests.iter().filter(|r| r.operation == "DoGet").count(),
            3
        );
        assert!(requests
            .iter()
            .filter(|r| r.operation == "DoGet")
            .all(|r| r.ticket_ids != submissions[0].ticket_ids));
        for request in requests {
            assert_eq!(
                request.headers.get("authorization").map(String::as_str),
                Some("Bearer fixture-token")
            );
            assert_eq!(
                request.headers.get("x-fixture-scope").map(String::as_str),
                Some("test")
            );
        }
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn query_aggregates_and_inner_limits_keep_their_scope() -> Result<()> {
    for (query, outer_limit, expected) in [
        ("SELECT COUNT(*) AS n FROM t", None, vec![Some(5)]),
        (
            "SELECT a FROM t ORDER BY a DESC LIMIT 3",
            Some(2),
            vec![Some(5), Some(4)],
        ),
        ("SELECT a FROM t LIMIT 3", Some(0), vec![]),
    ] {
        let fixture = SqlProviderFixture::start().await?;
        let table = factory()
            .open_table(
                fixture.endpoint(),
                FlightSqlSource::query(query)?,
                options(),
            )
            .await?;
        assert_eq!(
            fixture.requests()[0].sql,
            Some(format!("SELECT * FROM ({query}) LIMIT 1"))
        );
        assert!(table.statistics().is_none());
        let ctx = SessionContext::new();
        let plan = table.scan(&ctx.state(), None, &[], outer_limit).await?;
        let suffix = outer_limit
            .map(|n| format!(" LIMIT {n}"))
            .unwrap_or_default();
        assert_eq!(
            fixture.requests()[1].sql,
            Some(format!("SELECT * FROM ({query}){suffix}"))
        );
        assert_eq!(
            int_values(&collect(plan, ctx.task_ctx()).await?, 0),
            expected
        );
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn empty_tables_still_have_usable_schema() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("empty")?,
            options(),
        )
        .await?;
    assert_eq!(table.schema().fields().len(), 3);
    let ctx = SessionContext::new();
    for projection in [None, Some(vec![])] {
        let plan = table
            .scan(&ctx.state(), projection.as_ref(), &[], None)
            .await?;
        assert_eq!(
            plan.schema().fields().len(),
            if projection.is_some() { 0 } else { 3 }
        );
        let batches = collect(plan, ctx.task_ctx()).await?;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }
    fixture.shutdown().await
}

#[derive(Debug, Default)]
struct UnexpectedChannel(AtomicUsize);

#[async_trait]
impl FlightChannelProvider for UnexpectedChannel {
    async fn channel(&self, _endpoint: &str) -> Result<Channel> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Err(DataFusionError::Execution(
            "unexpected channel acquisition".into(),
        ))
    }
}

#[tokio::test]
async fn invalid_sources_and_conflicting_query_are_rejected_before_io() {
    let channel = Arc::new(UnexpectedChannel::default());
    let factory = factory().with_channel_provider(channel.clone());
    for source in [
        FlightSqlSource::Table(" ".into()),
        FlightSqlSource::Table("a\0b".into()),
        FlightSqlSource::Query("".into()),
        FlightSqlSource::Query("DELETE FROM t".into()),
        FlightSqlSource::Query("SELECT 1; SELECT 2".into()),
    ] {
        let error = factory
            .open_table("http://unused.invalid", source, options())
            .await
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)), "{error}");
    }
    let mut conflicting = options();
    conflicting.insert(QUERY.into(), "SELECT 1".into());
    let error = factory
        .open_table(
            "http://unused.invalid",
            FlightSqlSource::table("t").unwrap(),
            conflicting,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, DataFusionError::Plan(_)), "{error}");
    assert_eq!(channel.0.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn discovery_rejects_unusable_and_ambiguous_schemas() -> Result<()> {
    for response in [
        MetadataResponse::MissingSchema,
        MetadataResponse::MalformedSchema,
        MetadataResponse::Schema(Arc::new(Schema::empty())),
        MetadataResponse::Schema(Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("a", DataType::Int64, false),
        ]))),
    ] {
        let fixture = SqlProviderFixture::start().await?;
        fixture.set_next_metadata(response);
        assert!(factory()
            .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
            .await
            .is_err());
        assert_eq!(fixture.requests().len(), 1);
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn scan_schema_drift_is_rejected_before_fetching() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
        .await?;
    let fields = table.schema().fields().to_vec();
    let mut changed_name = fields.clone();
    changed_name[0] = Arc::new(changed_name[0].as_ref().clone().with_name("renamed"));
    let mut changed_type = fields.clone();
    changed_type[0] = Arc::new(
        changed_type[0]
            .as_ref()
            .clone()
            .with_data_type(DataType::Int32),
    );
    let mut changed_nullable = fields.clone();
    changed_nullable[0] = Arc::new(changed_nullable[0].as_ref().clone().with_nullable(true));
    let ctx = SessionContext::new();
    for schema in [
        Schema::new(changed_name),
        Schema::new(changed_type),
        Schema::new(changed_nullable),
        Schema::new(fields[..2].to_vec()),
    ] {
        fixture.set_next_metadata(MetadataResponse::Schema(Arc::new(schema)));
        let error = table.scan(&ctx.state(), None, &[], None).await.unwrap_err();
        assert!(error.to_string().contains("schema"), "{error}");
    }
    assert!(fixture
        .requests()
        .iter()
        .all(|r| r.operation == "GetFlightInfo"));
    fixture.shutdown().await
}

#[tokio::test]
async fn custom_metadata_differences_preserve_the_canonical_output() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
        .await?;
    let metadata = HashMap::from([("fixture".into(), "scan-only".into())]);
    let fields = table
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone().with_metadata(metadata.clone()))
        .collect::<Vec<_>>();
    fixture.set_next_metadata(MetadataResponse::Schema(Arc::new(
        Schema::new_with_metadata(fields, metadata),
    )));
    let ctx = SessionContext::new();
    let plan = table.scan(&ctx.state(), None, &[], None).await?;
    assert_eq!(plan.schema(), table.schema());
    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
    fixture.shutdown().await
}

#[tokio::test]
async fn scan_accepts_stronger_nullability() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    fixture.set_next_metadata(MetadataResponse::Schema(Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("c", DataType::Int64, true),
    ]))));
    let table = factory()
        .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
        .await?;
    let ctx = SessionContext::new();
    ctx.register_table("remote", Arc::new(table))?;
    let batches = ctx.sql("SELECT a FROM remote").await?.collect().await?;
    assert_eq!(
        int_values(&batches, 0),
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
    );
    assert!(batches
        .iter()
        .all(|batch| batch.schema().field(0).is_nullable()));
    fixture.shutdown().await
}

fn auth_options() -> HashMap<String, String> {
    let mut options = options();
    options.remove("flight.sql.header.authorization");
    options.insert(USERNAME.into(), "fixture".into());
    options.insert(PASSWORD.into(), "test".into());
    options
}

fn rpc_count(requests: &[RecordedRequest], operation: &str) -> usize {
    requests
        .iter()
        .filter(|request| request.operation == operation)
        .count()
}

#[derive(Debug)]
struct RecordingChannels {
    channel: Channel,
    delay: Duration,
    rejected: Vec<String>,
    requests: Mutex<Vec<String>>,
}

impl RecordingChannels {
    async fn connect(endpoint: &str) -> Result<Self> {
        Ok(Self {
            channel: Channel::from_shared(endpoint.to_owned())
                .map_err(|error| DataFusionError::External(Box::new(error)))?
                .connect()
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?,
            delay: Duration::ZERO,
            rejected: vec![],
            requests: Mutex::new(vec![]),
        })
    }
}

#[async_trait]
impl FlightChannelProvider for RecordingChannels {
    async fn channel(&self, endpoint: &str) -> Result<Channel> {
        self.requests.lock().unwrap().push(endpoint.to_owned());
        tokio::time::sleep(self.delay).await;
        if self.rejected.iter().any(|rejected| rejected == endpoint) {
            return Err(DataFusionError::Execution(format!(
                "fixture channel unavailable: {endpoint}"
            )));
        }
        Ok(self.channel.clone())
    }
}

fn round_trip(
    plan: Arc<dyn ExecutionPlan>,
    context: &SessionContext,
    channels: Arc<dyn FlightChannelProvider>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let codec = FlightPhysicalCodec::default();
    let mut bytes = vec![];
    codec.try_encode(plan, &mut bytes)?;
    let worker = codec.with_channel_provider(channels);
    let decoded = worker.try_decode(&bytes, &[], context.task_ctx().as_ref())?;
    let mut worker_bytes = vec![];
    worker.try_encode(decoded.clone(), &mut worker_bytes)?;
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
        serde_json::from_slice::<serde_json::Value>(&worker_bytes).unwrap()
    );
    Ok(decoded)
}

#[tokio::test]
async fn sql_rpc_headers_and_limits() -> Result<()> {
    const ORIGIN: &str = "http://planner.invalid";
    for persistent in [false, true] {
        for unbounded in [false, true] {
            let fixture = SqlProviderFixture::start().await?;
            fixture.require_ticket_auth();
            let channels = Arc::new(RecordingChannels::connect(fixture.endpoint()).await?);
            let request_options = auth_options();
            let original_options = request_options.clone();
            let table = FlightSqlTableFactory::new(
                FlightSqlDriver::new().with_persistent_headers(persistent),
            )
            .with_channel_provider(channels.clone())
            .with_options(
                FlightFactoryOptions::new()
                    .with_size_limits(SizeLimits::new(32000, 32000))
                    .with_properties(
                        FlightProperties::new()
                            .with_unbounded_streams(unbounded)
                            .with_reusable_flight_info(true)
                            .with_grpc_headers(HashMap::from([
                                ("x-fixture-scope".into(), "default".into()),
                                ("x-fixture-default".into(), "retained".into()),
                            ])),
                    ),
            )
            .open_table(
                ORIGIN,
                FlightSqlSource::table("t")?,
                request_options.clone(),
            )
            .await?;
            let context = SessionContext::new();
            for (column, expected) in [
                (0, vec![Some(1), Some(2), Some(3), Some(4), Some(5)]),
                (2, vec![Some(10), None, Some(30), Some(40), Some(50)]),
            ] {
                let plan = table
                    .scan(&context.state(), Some(&vec![column]), &[], None)
                    .await?;
                assert_eq!(
                    matches!(plan.properties().boundedness, Boundedness::Unbounded { .. }),
                    unbounded
                );
                assert_eq!(
                    int_values(&collect(plan, context.task_ctx()).await?, 0),
                    expected
                );
            }
            let requests = fixture.requests();
            assert_eq!(rpc_count(&requests, "Handshake"), 3);
            assert_eq!(rpc_count(&requests, "GetFlightInfo"), 3);
            assert_eq!(rpc_count(&requests, "DoGet"), 2);
            for request in requests
                .iter()
                .filter(|request| request.operation != "DoGet")
            {
                assert_eq!(request.headers["x-fixture-scope"], "test");
                assert_eq!(request.headers["x-fixture-default"], "retained");
                if request.operation == "Handshake" {
                    assert_eq!(request.headers["authorization"], "Basic Zml4dHVyZTp0ZXN0");
                }
            }
            let submissions = requests
                .iter()
                .filter(|request| request.operation == "GetFlightInfo")
                .collect::<Vec<_>>();
            let tokens = submissions
                .iter()
                .map(|request| request.headers["authorization"].as_str())
                .collect::<std::collections::HashSet<_>>();
            assert_eq!(tokens.len(), 3);
            for (scan, fetched) in submissions[1..].iter().zip(
                requests
                    .iter()
                    .filter(|request| request.operation == "DoGet"),
            ) {
                assert_eq!(scan.ticket_ids, fetched.ticket_ids);
                assert_eq!(
                    fetched.headers["authorization"],
                    scan.headers["authorization"]
                );
                assert_eq!(
                    fetched.headers.get("x-fixture-scope").map(String::as_str),
                    persistent.then_some("test")
                );
                assert_eq!(
                    fetched.headers.get("x-fixture-default").map(String::as_str),
                    persistent.then_some("retained")
                );
            }
            assert_eq!(*channels.requests.lock().unwrap(), vec![ORIGIN; 3]);
            assert_eq!(request_options, original_options);
            fixture.shutdown().await?;
        }
    }
    Ok(())
}

fn sql_request_size(sql: &str) -> usize {
    let command = CommandStatementQuery {
        query: sql.into(),
        transaction_id: None,
    };
    FlightDescriptor::new_cmd(command.as_any().encode_to_vec()).encoded_len()
}

#[tokio::test]
async fn sql_metadata_message_limits_cover_discovery_and_scan() -> Result<()> {
    let discovery_size = sql_request_size("SELECT * FROM \"t\" LIMIT 1");
    let scan_size = sql_request_size("SELECT \"a\", \"b\", \"c\" FROM \"t\"");
    for limit in [discovery_size - 1, discovery_size, scan_size - 1, scan_size] {
        let fixture = SqlProviderFixture::start().await?;
        let result = factory()
            .with_options(
                FlightFactoryOptions::new().with_size_limits(SizeLimits::new(limit, usize::MAX)),
            )
            .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
            .await;
        if limit < discovery_size {
            assert!(result.is_err());
            assert!(fixture.requests().is_empty());
        } else {
            let table = result?;
            let context = SessionContext::new();
            let result = table
                .scan(&context.state(), Some(&vec![0, 1, 2]), &[], None)
                .await;
            if limit < scan_size {
                assert!(result.is_err());
                assert_eq!(rpc_count(&fixture.requests(), "GetFlightInfo"), 1);
            } else {
                let batches = collect(result?, context.task_ctx()).await?;
                assert_eq!(
                    int_values(&batches, 0),
                    vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
                );
            }
        }
        fixture.shutdown().await?;
    }
    for discovery in [true, false] {
        let fixture = SqlProviderFixture::start().await?;
        let factory = factory().with_options(
            FlightFactoryOptions::new().with_size_limits(SizeLimits::new(usize::MAX, 1024)),
        );
        let oversized = Submission {
            metadata_padding: 4096,
            ..Default::default()
        };
        let error = if discovery {
            fixture.set_next_submission(oversized);
            factory
                .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
                .await
                .unwrap_err()
        } else {
            let table = factory
                .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
                .await?;
            fixture.set_next_submission(oversized);
            table
                .scan(&SessionContext::new().state(), None, &[], None)
                .await
                .unwrap_err()
        };
        assert!(
            error.to_string().contains("limit is: 1024 bytes"),
            "{error}"
        );
        assert_eq!(rpc_count(&fixture.requests(), "DoGet"), 0);
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn sql_execution_message_limits_survive_codec() -> Result<()> {
    for encoding in [true, false] {
        let server = SessionContext::new();
        server.register_batch(
            "t",
            RecordBatch::try_from_iter([(
                "a",
                Arc::new(Int64Array::from_iter_values(0..2048)) as _,
            )])?,
        )?;
        let fixture = SqlProviderFixture::start_with_context(server).await?;
        let table = factory()
            .with_options(FlightFactoryOptions::new().with_size_limits(if encoding {
                SizeLimits::new(1024, usize::MAX)
            } else {
                SizeLimits::new(usize::MAX, 1024)
            }))
            .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
            .await?;
        if encoding {
            fixture.set_next_submission(Submission {
                ticket_padding: 4096,
                ..Default::default()
            });
        }
        let context = SessionContext::new();
        let plan = table.scan(&context.state(), None, &[], None).await?;
        let channels = Arc::new(RecordingChannels::connect(fixture.endpoint()).await?);
        let decoded = round_trip(plan, &context, channels)?;
        let error = collect(decoded, context.task_ctx()).await.unwrap_err();
        if !encoding {
            assert!(
                error.to_string().contains("limit is: 1024 bytes"),
                "{error}"
            );
        }
        let requests = fixture.requests();
        assert_eq!(rpc_count(&requests, "GetFlightInfo"), 2);
        assert_eq!(rpc_count(&requests, "DoGet"), usize::from(!encoding));
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn sql_handshake_message_limits_cover_discovery_and_scan() -> Result<()> {
    for scan_failure in [false, true] {
        let fixture = SqlProviderFixture::start().await?;
        fixture.set_handshake_payload_size(4096);
        factory()
            .open_table(
                fixture.endpoint(),
                FlightSqlSource::table("t")?,
                auth_options(),
            )
            .await?;
        fixture.set_handshake_payload_size(0);
        let factory = factory().with_options(
            FlightFactoryOptions::new().with_size_limits(SizeLimits::new(usize::MAX, 1024)),
        );
        let table = if scan_failure {
            Some(
                factory
                    .open_table(
                        fixture.endpoint(),
                        FlightSqlSource::table("t")?,
                        auth_options(),
                    )
                    .await?,
            )
        } else {
            None
        };
        fixture.set_handshake_payload_size(4096);
        let error = if let Some(table) = table {
            table
                .scan(&SessionContext::new().state(), None, &[], None)
                .await
                .unwrap_err()
        } else {
            factory
                .open_table(
                    fixture.endpoint(),
                    FlightSqlSource::table("t")?,
                    auth_options(),
                )
                .await
                .unwrap_err()
        };
        assert!(matches!(&error, DataFusionError::External(_)), "{error}");
        let requests = fixture.requests();
        assert_eq!(
            rpc_count(&requests, "GetFlightInfo"),
            1 + usize::from(scan_failure)
        );
        assert_eq!(rpc_count(&requests, "DoGet"), 0);
        fixture.shutdown().await?;
    }
    Ok(())
}

#[rstest]
#[case::acquisition(2000, 0, 0)]
#[case::handshake(0, 2000, 0)]
#[case::query(0, 0, 2000)]
#[case::acquisition_and_auth(600, 600, 0)]
#[case::auth_and_query(0, 600, 600)]
#[case::all_phases(400, 400, 400)]
#[tokio::test]
async fn sql_opening_has_one_deadline(
    #[case] acquisition_ms: u64,
    #[case] handshake_ms: u64,
    #[case] query_ms: u64,
) -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    fixture.set_handshake(Duration::from_millis(handshake_ms), None);
    fixture.set_next_submission(Submission {
        delay: Duration::from_millis(query_ms),
        ..Default::default()
    });
    let mut channels = RecordingChannels::connect(fixture.endpoint()).await?;
    channels.delay = Duration::from_millis(acquisition_ms);
    let started = tokio::time::Instant::now();
    let error = factory()
        .with_channel_provider(Arc::new(channels))
        .with_options(FlightFactoryOptions::new().with_metadata_timeout(Duration::from_secs(1)))
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("t")?,
            auth_options(),
        )
        .await
        .unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(matches!(error, DataFusionError::Execution(message)
        if message.contains("timed out") && message.contains(fixture.endpoint()) && message.contains("1s")));
    assert_eq!(rpc_count(&fixture.requests(), "DoGet"), 0);
    fixture.shutdown().await
}

#[tokio::test]
async fn sql_scan_has_fresh_deadline() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    fixture.require_ticket_auth();
    fixture.set_next_submission(Submission {
        delay: Duration::from_millis(30),
        ..Default::default()
    });
    let channels = Arc::new(RecordingChannels::connect(fixture.endpoint()).await?);
    let table = factory()
        .with_channel_provider(channels.clone())
        .with_options(FlightFactoryOptions::new().with_metadata_timeout(Duration::from_millis(250)))
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("t")?,
            auth_options(),
        )
        .await?;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let context = SessionContext::new();
    fixture.set_next_submission(Submission {
        delay: Duration::from_millis(500),
        ..Default::default()
    });
    let error = table
        .scan(&context.state(), None, &[], None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error}");
    fixture.set_next_submission(Submission {
        delay: Duration::from_millis(30),
        do_get_delay: Duration::from_millis(300),
        ..Default::default()
    });
    let plan = table.scan(&context.state(), None, &[], None).await?;
    let batches = collect(plan, context.task_ctx()).await?;
    assert_eq!(
        int_values(&batches, 0),
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
    );
    let requests = fixture.requests();
    assert_eq!(rpc_count(&requests, "Handshake"), 3);
    assert_eq!(rpc_count(&requests, "GetFlightInfo"), 3);
    assert_eq!(rpc_count(&requests, "DoGet"), 1);
    let last_scan = requests
        .iter()
        .rev()
        .find(|request| request.operation == "GetFlightInfo")
        .unwrap();
    let fetched = requests
        .iter()
        .find(|request| request.operation == "DoGet")
        .unwrap();
    assert_eq!(fetched.ticket_ids, last_scan.ticket_ids);
    assert_eq!(
        fetched.headers["authorization"],
        last_scan.headers["authorization"]
    );
    assert_eq!(channels.requests.lock().unwrap().len(), 2);
    fixture.shutdown().await
}

#[tokio::test]
async fn sql_authentication_and_permission_errors_are_explicit() -> Result<()> {
    for scan_failure in [false, true] {
        for handshake_failure in [false, true] {
            let fixture = SqlProviderFixture::start().await?;
            let table = if scan_failure {
                Some(
                    factory()
                        .open_table(
                            fixture.endpoint(),
                            FlightSqlSource::table("t")?,
                            auth_options(),
                        )
                        .await?,
                )
            } else {
                None
            };
            let marker = if handshake_failure {
                fixture.set_handshake(
                    Duration::ZERO,
                    Some(Status::unauthenticated("fixture auth rejected")),
                );
                "fixture auth rejected"
            } else {
                fixture.set_next_submission(Submission {
                    error: Some(Status::permission_denied("fixture query denied")),
                    ..Default::default()
                });
                "fixture query denied"
            };
            let error = if let Some(table) = table {
                table
                    .scan(&SessionContext::new().state(), None, &[], None)
                    .await
                    .unwrap_err()
            } else {
                factory()
                    .open_table(
                        fixture.endpoint(),
                        FlightSqlSource::table("t")?,
                        auth_options(),
                    )
                    .await
                    .unwrap_err()
            };
            assert!(error.to_string().contains(marker), "{error}");
            let requests = fixture.requests();
            assert_eq!(
                rpc_count(&requests, "GetFlightInfo"),
                usize::from(scan_failure) + usize::from(!handshake_failure)
            );
            assert_eq!(rpc_count(&requests, "DoGet"), 0);
            fixture.shutdown().await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn sql_scan_rejects_invalid_schema_before_execution() -> Result<()> {
    for response in [
        MetadataResponse::MissingSchema,
        MetadataResponse::MalformedSchema,
        MetadataResponse::Schema(Arc::new(Schema::empty())),
    ] {
        let fixture = SqlProviderFixture::start().await?;
        let table = factory()
            .open_table(fixture.endpoint(), FlightSqlSource::table("t")?, options())
            .await?;
        fixture.set_next_metadata(response);
        assert!(table
            .scan(&SessionContext::new().state(), None, &[], None)
            .await
            .is_err());
        assert_eq!(rpc_count(&fixture.requests(), "GetFlightInfo"), 2);
        assert_eq!(rpc_count(&fixture.requests(), "DoGet"), 0);
        fixture.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn sql_expired_tickets_are_not_replayed() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("t")?,
            auth_options(),
        )
        .await?;
    fixture.set_next_submission(Submission {
        expired: true,
        ..Default::default()
    });
    let context = SessionContext::new();
    let plan = table.scan(&context.state(), None, &[], None).await?;
    let channels = Arc::new(RecordingChannels::connect(fixture.endpoint()).await?);
    let decoded = round_trip(plan, &context, channels)?;
    let before = fixture.requests();
    let error = collect(decoded, context.task_ctx()).await.unwrap_err();
    assert!(
        error.to_string().contains("fixture ticket expired"),
        "{error}"
    );
    let after = fixture.requests();
    assert_eq!(
        rpc_count(&after, "Handshake"),
        rpc_count(&before, "Handshake")
    );
    assert_eq!(
        rpc_count(&after, "GetFlightInfo"),
        rpc_count(&before, "GetFlightInfo")
    );
    assert_eq!(rpc_count(&after, "DoGet"), 1);
    assert_eq!(
        after.last().unwrap().ticket_ids,
        before.last().unwrap().ticket_ids
    );
    fixture.shutdown().await
}

#[tokio::test]
async fn sql_midstream_failures_are_not_replayed() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("t")?,
            auth_options(),
        )
        .await?;
    let gate = Arc::new(RequestGate::default());
    fixture.set_next_submission(Submission {
        batch_size: Some(2),
        stream_failure: Some(gate.clone()),
        ..Default::default()
    });
    let context = SessionContext::new();
    let plan = table
        .scan(&context.state(), Some(&vec![0]), &[], None)
        .await?;
    let before = fixture.requests();
    let mut stream = plan.execute(0, context.task_ctx())?;
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))?
        .unwrap()?;
    assert_eq!(int_values(&[first], 0), vec![Some(1), Some(2)]);
    gate.wait_entered().await?;
    gate.release();
    let error = stream.try_collect::<Vec<_>>().await.unwrap_err();
    assert!(
        error.to_string().contains("fixture midstream failure"),
        "{error}"
    );
    let after = fixture.requests();
    assert_eq!(
        rpc_count(&after, "Handshake"),
        rpc_count(&before, "Handshake")
    );
    assert_eq!(
        rpc_count(&after, "GetFlightInfo"),
        rpc_count(&before, "GetFlightInfo")
    );
    assert_eq!(rpc_count(&after, "DoGet"), 1);
    fixture.shutdown().await
}

#[tokio::test]
async fn concurrent_scans_do_not_share_state() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    fixture.require_ticket_auth();
    let request_options = auth_options();
    let table = factory()
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("t")?,
            request_options.clone(),
        )
        .await?;
    let canonical_schema = table.schema();
    let first_sql = "SELECT \"a\" FROM \"t\" LIMIT 2";
    let second_sql = "SELECT \"c\" FROM \"t\" LIMIT 3";
    let first_gate = Arc::new(RequestGate::default());
    let second_gate = Arc::new(RequestGate::default());
    for (sql, gate) in [
        (first_sql, first_gate.clone()),
        (second_sql, second_gate.clone()),
    ] {
        fixture.set_submission(
            sql,
            Submission {
                gate: Some(gate),
                ..Default::default()
            },
        );
    }
    let context = SessionContext::new();
    let state = context.state();
    let first_projection = vec![0];
    let second_projection = vec![2];
    let second_completed = tokio::sync::Notify::new();
    let second_scan = async {
        first_gate.wait_entered().await?;
        let result = table
            .scan(&state, Some(&second_projection), &[], Some(3))
            .await;
        second_completed.notify_one();
        result
    };
    let control = async {
        second_gate.wait_entered().await?;
        second_gate.release();
        second_completed.notified().await;
        first_gate.release();
        Ok::<_, DataFusionError>(())
    };
    let (first, second, control) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(
            table.scan(&state, Some(&first_projection), &[], Some(2)),
            second_scan,
            control,
        )
    })
    .await
    .map_err(|error| DataFusionError::External(Box::new(error)))?;
    control?;
    let first = first?;
    let second = second?;
    assert_eq!(&fixture.completed_sql()[1..], &[second_sql, first_sql]);
    assert_eq!(first.schema().fields().len(), 1);
    assert_eq!(first.schema().field(0), canonical_schema.field(0));
    assert_eq!(second.schema().fields().len(), 1);
    assert_eq!(second.schema().field(0), canonical_schema.field(2));
    assert_eq!(
        int_values(&collect(first, context.task_ctx()).await?, 0),
        vec![Some(1), Some(2)]
    );
    assert_eq!(
        int_values(&collect(second, context.task_ctx()).await?, 0),
        vec![Some(10), None, Some(30)]
    );
    let repeated = table
        .scan(&state, Some(&first_projection), &[], Some(1))
        .await?;
    assert_eq!(
        int_values(&collect(repeated, context.task_ctx()).await?, 0),
        vec![Some(1)]
    );
    assert_eq!(table.schema(), canonical_schema);
    assert_eq!(request_options, auth_options());
    let requests = fixture.requests();
    let submissions = requests
        .iter()
        .filter(|request| request.operation == "GetFlightInfo")
        .collect::<Vec<_>>();
    assert_eq!(submissions.len(), 4);
    let tokens = submissions
        .iter()
        .map(|request| request.headers["authorization"].as_str())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(tokens.len(), 4);
    let tickets = submissions
        .iter()
        .flat_map(|request| request.ticket_ids.iter())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(tickets.len(), 4);
    assert_eq!(submissions[1].sql.as_deref(), Some(first_sql));
    assert_eq!(submissions[2].sql.as_deref(), Some(second_sql));
    assert_eq!(
        submissions[3].sql.as_deref(),
        Some("SELECT \"a\" FROM \"t\" LIMIT 1")
    );
    assert!(requests
        .iter()
        .all(|request| request.headers["x-fixture-scope"] == "test"));
    for fetched in requests
        .iter()
        .filter(|request| request.operation == "DoGet")
    {
        let scan = submissions
            .iter()
            .find(|request| request.ticket_ids == fetched.ticket_ids)
            .unwrap();
        assert_ne!(scan.ticket_ids, submissions[0].ticket_ids);
        assert_eq!(
            fetched.headers["authorization"],
            scan.headers["authorization"]
        );
    }
    fixture.shutdown().await
}

#[tokio::test]
async fn decoded_plan_uses_only_saved_tickets() -> Result<()> {
    const ORIGIN: &str = "http://planner.invalid";
    const UNAVAILABLE: &str = "http://unavailable.invalid";
    const ALTERNATE: &str = "http://alternate.invalid";
    for (name, rows) in [("t", 4), ("empty", 0)] {
        for projection in [vec![0], vec![]] {
            let fixture = SqlProviderFixture::start().await?;
            fixture.require_ticket_auth();
            let planner = Arc::new(RecordingChannels::connect(fixture.endpoint()).await?);
            let mut worker = RecordingChannels::connect(fixture.endpoint()).await?;
            worker.rejected.push(UNAVAILABLE.into());
            let worker = Arc::new(worker);
            let table = factory()
                .with_channel_provider(planner.clone())
                .open_table(ORIGIN, FlightSqlSource::table(name)?, auth_options())
                .await?;
            fixture.set_next_submission(Submission {
                locations: vec![
                    vec![],
                    vec!["arrow-flight-reuse-connection://?".into()],
                    vec![UNAVAILABLE.into(), ALTERNATE.into()],
                ],
                batch_size: Some(1),
                empty_batches: true,
                ..Default::default()
            });
            let context = SessionContext::new();
            let plan = table
                .scan(&context.state(), Some(&projection), &[], Some(4))
                .await?;
            let decoded = round_trip(plan, &context, worker.clone())?;
            assert_eq!(decoded.output_partitioning().partition_count(), 3);
            assert_eq!(decoded.schema().fields().len(), projection.len());
            let before = fixture.requests();
            assert_eq!(rpc_count(&before, "Handshake"), 2);
            assert_eq!(rpc_count(&before, "GetFlightInfo"), 2);
            assert_eq!(rpc_count(&before, "DoGet"), 0);
            assert!(worker.requests.lock().unwrap().is_empty());
            let scan = before.last().unwrap();
            assert!(scan.sql.as_deref().unwrap().ends_with(" LIMIT 4"));
            let batches = collect(decoded, context.task_ctx()).await?;
            assert_eq!(
                batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                rows
            );
            assert!(batches
                .iter()
                .all(|batch| batch.num_columns() == projection.len()));
            if !projection.is_empty() {
                let mut values = int_values(&batches, 0);
                values.sort();
                assert_eq!(
                    values,
                    if rows == 0 {
                        vec![]
                    } else {
                        vec![Some(1), Some(2), Some(3), Some(4)]
                    }
                );
            }
            let after = fixture.requests();
            assert_eq!(
                rpc_count(&after, "Handshake"),
                rpc_count(&before, "Handshake")
            );
            assert_eq!(
                rpc_count(&after, "GetFlightInfo"),
                rpc_count(&before, "GetFlightInfo")
            );
            let fetched = after
                .iter()
                .filter(|request| request.operation == "DoGet")
                .collect::<Vec<_>>();
            assert_eq!(fetched.len(), 3);
            let mut fetched_ids = fetched
                .iter()
                .flat_map(|request| request.ticket_ids.clone())
                .collect::<Vec<_>>();
            fetched_ids.sort();
            let mut issued_ids = scan.ticket_ids.clone();
            issued_ids.sort();
            assert_eq!(fetched_ids, issued_ids);
            for request in fetched {
                assert_eq!(
                    request.headers["authorization"],
                    scan.headers["authorization"]
                );
                assert_eq!(request.headers["x-fixture-scope"], "test");
            }
            assert_eq!(*planner.requests.lock().unwrap(), vec![ORIGIN]);
            let mut worker_requests = worker.requests.lock().unwrap().clone();
            worker_requests.sort();
            let mut expected = vec![ORIGIN, ORIGIN, UNAVAILABLE, ALTERNATE];
            expected.sort();
            assert_eq!(worker_requests, expected);
            fixture.shutdown().await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn explain_plans_without_fetching_tickets() -> Result<()> {
    let fixture = SqlProviderFixture::start().await?;
    let table = factory()
        .open_table(
            fixture.endpoint(),
            FlightSqlSource::table("t")?,
            auth_options(),
        )
        .await?;
    let context = SessionContext::new();
    context.register_table("remote", Arc::new(table))?;
    context
        .sql("EXPLAIN SELECT a FROM remote WHERE a > 1 LIMIT 1")
        .await?
        .collect()
        .await?;
    let explained = fixture.requests();
    assert!(rpc_count(&explained, "GetFlightInfo") >= 2);
    assert_eq!(rpc_count(&explained, "DoGet"), 0);
    let plan = context
        .sql("SELECT a FROM remote LIMIT 2")
        .await?
        .create_physical_plan()
        .await?;
    let before = fixture.requests();
    assert_eq!(rpc_count(&before, "DoGet"), 0);
    let batches = collect(plan, context.task_ctx()).await?;
    assert_eq!(int_values(&batches, 0), vec![Some(1), Some(2)]);
    let after = fixture.requests();
    assert_eq!(
        rpc_count(&after, "Handshake"),
        rpc_count(&before, "Handshake")
    );
    assert_eq!(
        rpc_count(&after, "GetFlightInfo"),
        rpc_count(&before, "GetFlightInfo")
    );
    assert_eq!(rpc_count(&after, "DoGet"), 1);
    assert_eq!(
        after.last().unwrap().ticket_ids,
        before.last().unwrap().ticket_ids
    );
    fixture.shutdown().await
}
