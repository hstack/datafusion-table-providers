use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::collect;
use datafusion::prelude::{col, lit, SessionConfig, SessionContext};
use datafusion_table_providers::flight::sql::{FlightSqlDriver, QUERY};
use datafusion_table_providers::flight::{
    FlightChannelProvider, FlightFactoryOptions, FlightProperties, FlightSqlSource,
    FlightSqlTableFactory, FlightSqlTableOptions,
};
use tonic::transport::Channel;

mod fixture;

use fixture::{MetadataResponse, SqlProviderFixture};

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
