use std::sync::Arc;

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, TicketStatementQuery};
use arrow_flight::{FlightDescriptor, Ticket};
use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::catalog::TableProvider;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion_table_providers::flight::sql::FlightSqlDriver;
use datafusion_table_providers::flight::{FlightProperties, FlightTableFactory, SizeLimits};
use futures::{stream, TryStreamExt};
use prost::Message;
use rstest::rstest;

use super::{batch, options, start_service, test_service, QUERY};

fn limits(encoding: bool, size: usize) -> SizeLimits {
    if encoding {
        SizeLimits::new(size, usize::MAX)
    } else {
        SizeLimits::new(usize::MAX, size)
    }
}

#[rstest]
#[case::encoding_at_limit(true, 0)]
#[case::encoding_over_limit(true, 1)]
#[case::decoding_at_limit(false, 0)]
#[case::decoding_over_limit(false, 1)]
#[test_log::test(tokio::test)]
async fn test_metadata_message_size_limits(#[case] encoding: bool, #[case] overflow: usize) {
    let service = test_service(batch(7), vec![vec![]]);
    let size = if encoding {
        let command = CommandStatementQuery {
            query: QUERY.into(),
            transaction_id: None,
        };
        FlightDescriptor::new_cmd(command.as_any().encode_to_vec()).encoded_len()
    } else {
        service.flight_info.encoded_len()
    };
    let limit = size - overflow;
    let (endpoint, _shutdown) = start_service(service).await;
    let driver = FlightSqlDriver::new().with_properties_template(
        FlightProperties::new().with_size_limits(limits(encoding, limit)),
    );
    let result = FlightTableFactory::new(Arc::new(driver))
        .open_table(endpoint, options())
        .await;
    if overflow == 0 {
        let table = result.unwrap();
        assert_eq!(table.schema(), batch(7).schema());
        let ctx = SessionContext::new();
        let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
        assert_eq!(plan.schema(), batch(7).schema());
    } else {
        let error = result.unwrap_err().to_string();
        // Tonic surfaces request encoding failures as HTTP/2 resets.
        if !encoding {
            assert!(
                error.contains(&format!("found {size} bytes, the limit is: {limit} bytes")),
                "{error}"
            );
        }
    }
}

#[rstest]
#[case::encoding_at_limit(true, 0)]
#[case::encoding_over_limit(true, 1)]
#[case::decoding_at_limit(false, 0)]
#[case::decoding_over_limit(false, 1)]
#[test_log::test(tokio::test)]
async fn test_do_get_message_size_limits(#[case] encoding: bool, #[case] overflow: usize) {
    let data = RecordBatch::try_new(
        batch(7).schema(),
        vec![Arc::new(Int64Array::from(vec![7; 1024]))],
    )
    .unwrap();
    let mut service = test_service(data.clone(), vec![vec![]]);
    let size = if encoding {
        let ticket = Ticket::new(
            TicketStatementQuery {
                statement_handle: vec![0; 1024].into(),
            }
            .as_any()
            .encode_to_vec(),
        );
        let size = ticket.encoded_len();
        service.flight_info.endpoint[0].ticket = Some(ticket);
        size
    } else {
        let messages: Vec<_> = FlightDataEncoderBuilder::default()
            .with_schema(data.schema())
            .build(stream::iter([Ok(data.clone())]))
            .try_collect()
            .await
            .unwrap();
        messages.iter().map(Message::encoded_len).max().unwrap()
    };
    let limit = size - overflow;
    let (endpoint, _shutdown) = start_service(service).await;
    let driver = FlightSqlDriver::new().with_properties_template(
        FlightProperties::new().with_size_limits(limits(encoding, limit)),
    );
    let table = FlightTableFactory::new(Arc::new(driver))
        .open_table(endpoint, options())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
    let result = collect(plan, ctx.task_ctx()).await;
    if overflow == 0 {
        assert_eq!(result.unwrap(), vec![data]);
    } else {
        let error = result.unwrap_err().to_string();
        if !encoding {
            assert!(
                error.contains(&format!("found {size} bytes, the limit is: {limit} bytes")),
                "{error}"
            );
        }
    }
}

#[test_log::test(tokio::test)]
async fn test_default_limits_allow_metadata_above_tonic_default() {
    let mut service = test_service(batch(7), vec![vec![]]);
    service.flight_info.app_metadata = vec![0; 4 * 1024 * 1024].into();
    let (endpoint, _shutdown) = start_service(service).await;
    let table = FlightTableFactory::new(Arc::new(FlightSqlDriver::new()))
        .open_table(endpoint, options())
        .await
        .unwrap();
    assert_eq!(table.schema(), batch(7).schema());
}
