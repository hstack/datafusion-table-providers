use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_flight::sql::{ProstMessageExt, TicketStatementQuery};
use arrow_flight::{FlightEndpoint, FlightInfo, Location, Ticket};
use async_trait::async_trait;
use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::common::{DataFusionError, Result};
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_table_providers::flight::codec::FlightPhysicalCodec;
use datafusion_table_providers::flight::sql::FlightSqlDriver;
use datafusion_table_providers::flight::{FlightChannelProvider, FlightTableFactory};
use prost::Message;
use rstest::rstest;
use tokio::sync::oneshot::{channel, Sender};
use tonic::transport::Channel;

use super::TestFlightSqlService;

const ORIGIN: &str = "http://origin.invalid";
const DATA_LOCATION: &str = "http://data.invalid";
const QUERY: &str = "SELECT id FROM test";

#[derive(Debug, Default)]
struct TestChannelProvider {
    channel: Option<Channel>,
    requests: Mutex<Vec<String>>,
}

impl TestChannelProvider {
    async fn connect(endpoint: &str) -> Arc<Self> {
        Arc::new(Self {
            channel: Some(
                Channel::from_shared(endpoint.to_owned())
                    .unwrap()
                    .connect()
                    .await
                    .unwrap(),
            ),
            ..Default::default()
        })
    }
}

#[async_trait]
impl FlightChannelProvider for TestChannelProvider {
    async fn channel(&self, endpoint: &str) -> Result<Channel> {
        self.requests.lock().unwrap().push(endpoint.to_owned());
        self.channel
            .clone()
            .ok_or_else(|| DataFusionError::Execution("channel acquisition failed".into()))
    }
}

fn batch(value: i64) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![value]))],
    )
    .unwrap()
}

async fn start_service(value: i64, locations: Vec<Vec<&str>>) -> (String, Sender<()>) {
    let partition_data = batch(value);
    let mut flight_info = FlightInfo::new()
        .try_with_schema(partition_data.schema().as_ref())
        .unwrap();
    let ticket = Ticket::new(TicketStatementQuery::default().as_any().encode_to_vec());
    for locations in locations {
        let mut endpoint = FlightEndpoint::new().with_ticket(ticket.clone());
        endpoint.location = locations
            .into_iter()
            .map(|uri| Location { uri: uri.into() })
            .collect();
        flight_info = flight_info.with_endpoint(endpoint);
    }
    let (tx, rx) = channel();
    let service = TestFlightSqlService {
        flight_info,
        partition_data,
        expected_handshake_headers: HashMap::new(),
        expected_flight_info_query: QUERY.into(),
        shutdown_sender: None,
    };
    let address = service.run_in_background(rx).await;
    (format!("http://{address}"), tx)
}

fn options() -> HashMap<String, String> {
    HashMap::from([
        ("flight.sql.query".into(), QUERY.into()),
        ("flight.sql.username".into(), "admin".into()),
        ("flight.sql.password".into(), "password".into()),
    ])
}

#[rstest]
#[case::origin(vec![vec![]], vec![ORIGIN])]
#[case::reuse(vec![vec!["arrow-flight-reuse-connection://?"]], vec![ORIGIN])]
#[case::location(vec![vec![DATA_LOCATION]], vec![DATA_LOCATION])]
#[case::partitions(vec![vec![], vec![DATA_LOCATION]], vec![ORIGIN, DATA_LOCATION])]
#[test_log::test(tokio::test)]
async fn test_channel_provider_for_metadata_and_execution(
    #[case] locations: Vec<Vec<&str>>,
    #[case] execution_locations: Vec<&str>,
) {
    let (endpoint, _shutdown) = start_service(7, locations).await;
    let provider = TestChannelProvider::connect(&endpoint).await;
    let factory = FlightTableFactory::new(Arc::new(FlightSqlDriver::new()))
        .with_channel_provider(provider.clone());
    let ctx = SessionContext::new();
    let mut expected_requests = vec![];
    for factory in [factory.clone(), factory] {
        let table = factory.open_table(ORIGIN, options()).await.unwrap();
        expected_requests.push(ORIGIN);
        for _ in 0..2 {
            let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
            assert_eq!(
                collect(plan, ctx.task_ctx()).await.unwrap(),
                vec![batch(7); execution_locations.len()]
            );
            expected_requests.extend_from_slice(&execution_locations);
        }
    }
    assert_eq!(*provider.requests.lock().unwrap(), expected_requests);
}

#[test_log::test(tokio::test)]
async fn test_codec_uses_worker_provider_without_changing_serialized_plan() {
    let (planner_endpoint, _planner_shutdown) = start_service(7, vec![vec![]]).await;
    let (worker_endpoint, _worker_shutdown) = start_service(9, vec![vec![]]).await;
    let planner = TestChannelProvider::connect(&planner_endpoint).await;
    let worker = TestChannelProvider::connect(&worker_endpoint).await;
    let factory = FlightTableFactory::new(Arc::new(FlightSqlDriver::new()))
        .with_channel_provider(planner.clone());
    let table = factory
        .open_table(&planner_endpoint, options())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
    let default_codec = FlightPhysicalCodec::default();
    let mut bytes = vec![];
    default_codec.try_encode(plan, &mut bytes).unwrap();
    let default_plan = default_codec
        .try_decode(&bytes, &[], ctx.task_ctx().as_ref())
        .unwrap();
    assert_eq!(
        collect(default_plan, ctx.task_ctx()).await.unwrap(),
        vec![batch(7)]
    );

    let planner_codec = default_codec.with_channel_provider(planner.clone());
    let worker_codec = planner_codec.clone().with_channel_provider(worker.clone());
    let worker_plan = worker_codec
        .try_decode(&bytes, &[], ctx.task_ctx().as_ref())
        .unwrap();
    let mut worker_bytes = vec![];
    worker_codec
        .try_encode(worker_plan.clone(), &mut worker_bytes)
        .unwrap();
    assert_eq!(bytes, worker_bytes);
    assert_eq!(
        collect(worker_plan, ctx.task_ctx()).await.unwrap(),
        vec![batch(9)]
    );
    let planner_plan = planner_codec
        .try_decode(&bytes, &[], ctx.task_ctx().as_ref())
        .unwrap();
    assert_eq!(
        collect(planner_plan, ctx.task_ctx()).await.unwrap(),
        vec![batch(7)]
    );
    assert_eq!(
        *planner.requests.lock().unwrap(),
        vec![planner_endpoint.clone(), planner_endpoint.clone()]
    );
    assert_eq!(*worker.requests.lock().unwrap(), vec![planner_endpoint]);
}

#[test_log::test(tokio::test)]
async fn test_channel_acquisition_errors_propagate() {
    let failing = Arc::new(TestChannelProvider::default());
    let driver = Arc::new(FlightSqlDriver::new());
    let factory = FlightTableFactory::new(driver.clone()).with_channel_provider(failing.clone());
    let error = factory.open_table(ORIGIN, options()).await.unwrap_err();
    assert!(matches!(error, DataFusionError::Execution(message)
        if message == "channel acquisition failed"));

    let (endpoint, _shutdown) = start_service(7, vec![vec![]]).await;
    let provider = TestChannelProvider::connect(&endpoint).await;
    let factory = FlightTableFactory::new(driver).with_channel_provider(provider);
    let table = factory.open_table(ORIGIN, options()).await.unwrap();
    let ctx = SessionContext::new();
    let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
    let codec = FlightPhysicalCodec::default().with_channel_provider(failing);
    let mut bytes = vec![];
    codec.try_encode(plan, &mut bytes).unwrap();
    let restored = codec
        .try_decode(&bytes, &[], ctx.task_ctx().as_ref())
        .unwrap();
    let error = collect(restored, ctx.task_ctx()).await.unwrap_err();
    assert!(error.to_string().contains("channel acquisition failed"));
}
