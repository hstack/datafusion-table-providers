use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::FlightInfo;
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::common::stats::Precision;
use datafusion::common::{DataFusionError, Result};
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::prelude::SessionContext;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_table_providers::flight::codec::FlightPhysicalCodec;
use datafusion_table_providers::flight::sql::{FlightSqlDriver, HEADER_PREFIX};
use datafusion_table_providers::flight::{
    FlightChannelProvider, FlightDriver, FlightFactoryOptions, FlightMetadata, FlightProperties,
    FlightTableFactory,
};
use rstest::rstest;
use tokio::sync::Notify;
use tokio::time::{sleep, Instant};
use tonic::transport::Channel;
use tonic::Status;

use super::{batch, options, start_service, test_service, AUTH_HEADER, BEARER_TOKEN};

const ORIGIN: &str = "http://origin.invalid";

#[derive(Debug)]
struct DelayedChannelProvider(Duration);

#[async_trait]
impl FlightChannelProvider for DelayedChannelProvider {
    async fn channel(&self, _endpoint: &str) -> Result<Channel> {
        sleep(self.0).await;
        Ok(Channel::from_static("http://127.0.0.1:1").connect_lazy())
    }
}

#[derive(Debug)]
struct DelayedDriver {
    delay: Mutex<Duration>,
    info: FlightInfo,
}

#[async_trait]
impl FlightDriver for DelayedDriver {
    async fn metadata(
        &self,
        _client: FlightServiceClient<Channel>,
        properties: FlightProperties,
        _options: &HashMap<String, String>,
    ) -> arrow_flight::error::Result<FlightMetadata> {
        let delay = *self.delay.lock().unwrap();
        sleep(delay).await;
        FlightMetadata::try_new(self.info.clone(), properties)
    }
}

fn factory(
    channel_delay: Duration,
    metadata_delay: Duration,
) -> (FlightTableFactory, Arc<DelayedDriver>) {
    let driver = Arc::new(DelayedDriver {
        delay: Mutex::new(metadata_delay),
        info: test_service(batch(7), vec![vec![]]).flight_info.clone(),
    });
    let factory = FlightTableFactory::new(driver.clone())
        .with_channel_provider(Arc::new(DelayedChannelProvider(channel_delay)));
    (factory, driver)
}

fn assert_timed_out(error: DataFusionError) {
    assert!(matches!(error, DataFusionError::Execution(message)
        if message.contains("timed out") && message.contains(ORIGIN) && message.contains("1s")));
}

#[rstest]
#[case::acquisition(Duration::from_secs(2), Duration::ZERO)]
#[case::metadata(Duration::ZERO, Duration::from_secs(2))]
#[case::combined(Duration::from_millis(600), Duration::from_millis(600))]
#[tokio::test(start_paused = true)]
async fn test_metadata_timeout_is_one_operation_budget(
    #[case] channel_delay: Duration,
    #[case] metadata_delay: Duration,
) {
    let (factory, _) = factory(channel_delay, metadata_delay);
    let factory = factory
        .with_options(FlightFactoryOptions::new().with_metadata_timeout(Duration::from_secs(1)));
    let started = Instant::now();
    let error = factory.open_table(ORIGIN, options()).await.unwrap_err();
    assert_eq!(started.elapsed(), Duration::from_secs(1));
    assert_timed_out(error);
}

#[tokio::test(start_paused = true)]
async fn test_refresh_gets_a_fresh_budget_and_recovers_after_timeout() {
    let (factory, driver) = factory(Duration::from_millis(600), Duration::ZERO);
    let table = factory
        .with_options(FlightFactoryOptions::new().with_metadata_timeout(Duration::from_secs(1)))
        .open_table(ORIGIN, options())
        .await
        .unwrap();
    sleep(Duration::from_secs(2)).await;
    *driver.delay.lock().unwrap() = Duration::from_secs(2);
    let ctx = SessionContext::new();
    let started = Instant::now();
    let error = table.scan(&ctx.state(), None, &[], None).await.unwrap_err();
    assert_eq!(started.elapsed(), Duration::from_secs(1));
    assert_timed_out(error);

    *driver.delay.lock().unwrap() = Duration::ZERO;
    let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
    assert_eq!(plan.schema(), batch(7).schema());
}

#[tokio::test(start_paused = true)]
async fn test_metadata_has_no_timeout_by_default() {
    let (factory, _) = factory(Duration::from_secs(2), Duration::from_secs(2));
    let started = Instant::now();
    let table = factory.open_table(ORIGIN, options()).await.unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(4));
    assert_eq!(table.schema(), batch(7).schema());
}

#[tokio::test(start_paused = true)]
async fn test_factory_property_defaults_are_used_for_reusable_metadata() {
    let (factory, driver) = factory(Duration::ZERO, Duration::ZERO);
    let table = factory
        .with_options(
            FlightFactoryOptions::new()
                .with_metadata_timeout(Duration::from_secs(1))
                .with_properties(
                    FlightProperties::new()
                        .with_reusable_flight_info(true)
                        .with_unbounded_streams(true),
                ),
        )
        .open_table(ORIGIN, options())
        .await
        .unwrap();
    *driver.delay.lock().unwrap() = Duration::from_secs(2);
    let ctx = SessionContext::new();
    let started = Instant::now();
    let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
    assert_eq!(started.elapsed(), Duration::ZERO);
    assert!(matches!(
        plan.properties().boundedness,
        Boundedness::Unbounded { .. }
    ));
}

#[derive(Debug)]
struct MetadataOnlyDriver {
    calls: AtomicUsize,
    info: FlightInfo,
}

#[async_trait]
impl FlightDriver for MetadataOnlyDriver {
    async fn metadata(
        &self,
        _client: FlightServiceClient<Channel>,
        properties: FlightProperties,
        _options: &HashMap<String, String>,
    ) -> arrow_flight::error::Result<FlightMetadata> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let info = self
            .info
            .clone()
            .with_total_records(17 + call as i64)
            .with_total_bytes(99 + call as i64);
        FlightMetadata::try_new(info, properties)
    }
}

#[rstest]
#[case::refresh(false)]
#[case::reuse(true)]
#[tokio::test]
async fn metadata_only_driver_reuses_only_explicit_metadata(#[case] reusable: bool) {
    let driver = Arc::new(MetadataOnlyDriver {
        calls: AtomicUsize::new(0),
        info: test_service(batch(7), vec![vec![]]).flight_info.clone(),
    });
    let table = FlightTableFactory::new(driver.clone())
        .with_channel_provider(Arc::new(DelayedChannelProvider(Duration::ZERO)))
        .with_options(
            FlightFactoryOptions::new()
                .with_properties(FlightProperties::new().with_reusable_flight_info(reusable)),
        )
        .open_table(ORIGIN, options())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    for _ in 0..2 {
        let plan = table.scan(&ctx.state(), None, &[], None).await.unwrap();
        assert_eq!(plan.schema(), batch(7).schema());
        assert!(matches!(
            plan.properties().boundedness,
            Boundedness::Bounded
        ));
        let stats = table.statistics().unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(17));
        assert_eq!(stats.total_byte_size, Precision::Exact(99));
    }
    assert_eq!(
        driver.calls.load(Ordering::SeqCst),
        if reusable { 1 } else { 3 }
    );
}

#[derive(Debug)]
struct PendingMetadataDriver {
    calls: AtomicUsize,
    info: FlightInfo,
    observed_options: Mutex<Vec<HashMap<String, String>>>,
    started: [Notify; 2],
    release: [Notify; 2],
}

#[async_trait]
impl FlightDriver for PendingMetadataDriver {
    async fn metadata(
        &self,
        _client: FlightServiceClient<Channel>,
        properties: FlightProperties,
        options: &HashMap<String, String>,
    ) -> arrow_flight::error::Result<FlightMetadata> {
        self.observed_options.lock().unwrap().push(options.clone());
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let mut info = self.info.clone();
        if call > 0 {
            let index = call - 1;
            self.started[index].notify_one();
            self.release[index].notified().await;
            info.endpoint[0].ticket = Some(arrow_flight::Ticket::new(vec![call as u8]));
        }
        let properties = properties
            .with_grpc_headers(HashMap::from([("x-response".into(), call.to_string())]))
            .with_unbounded_streams(call == 1);
        FlightMetadata::try_new(info, properties)
    }
}

#[tokio::test]
async fn concurrent_refreshes_keep_options_immutable() {
    let driver = Arc::new(PendingMetadataDriver {
        calls: AtomicUsize::new(0),
        info: test_service(batch(7), vec![vec![]]).flight_info.clone(),
        observed_options: Mutex::new(vec![]),
        started: [Notify::new(), Notify::new()],
        release: [Notify::new(), Notify::new()],
    });
    let original_options = options();
    let table = FlightTableFactory::new(driver.clone())
        .with_channel_provider(Arc::new(DelayedChannelProvider(Duration::ZERO)))
        .open_table(ORIGIN, original_options.clone())
        .await
        .unwrap();
    driver.observed_options.lock().unwrap().clear();
    let ctx = SessionContext::new();
    let state = ctx.state();
    let mut first = Box::pin(table.scan(&state, None, &[], None));
    let mut second = Box::pin(table.scan(&state, None, &[], None));
    tokio::select! {
        biased;
        _ = driver.started[0].notified() => {}
        _ = &mut first => panic!("first refresh completed before release"),
        _ = sleep(Duration::from_secs(5)) => panic!("first refresh did not start"),
    }
    tokio::select! {
        biased;
        _ = driver.started[1].notified() => {}
        _ = &mut second => panic!("second refresh completed before release"),
        _ = sleep(Duration::from_secs(5)) => panic!("second refresh did not start"),
    }
    driver.release[1].notify_one();
    let second_plan = tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .unwrap()
        .unwrap();
    driver.release[0].notify_one();
    let first_plan = tokio::time::timeout(Duration::from_secs(5), first)
        .await
        .unwrap()
        .unwrap();

    let codec = FlightPhysicalCodec::default();
    for (plan, call) in [(first_plan, 1), (second_plan, 2)] {
        assert_eq!(plan.schema(), batch(7).schema());
        assert_eq!(
            matches!(plan.properties().boundedness, Boundedness::Unbounded { .. }),
            call == 1
        );
        let mut bytes = vec![];
        codec.try_encode(plan, &mut bytes).unwrap();
        let config: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(config["partitions"][0]["ticket"], serde_json::json!([call]));
        assert_eq!(
            config["properties"]["grpc_headers"]["x-response"],
            call.to_string()
        );
    }
    let observed_options = driver.observed_options.lock().unwrap().clone();
    assert_eq!(observed_options, vec![original_options; 2]);
    assert_eq!(driver.calls.load(Ordering::SeqCst), 3);
}

#[test_log::test(tokio::test)]
async fn test_handshake_failure_is_not_ignored() {
    let mut service = test_service(batch(7), vec![vec![]]);
    service.handshake_error = Some(Status::unauthenticated("handshake rejected"));
    let (endpoint, _shutdown) = start_service(service).await;
    let mut options = options();
    options.insert(format!("{HEADER_PREFIX}{AUTH_HEADER}"), BEARER_TOKEN.into());
    let error = FlightTableFactory::new(Arc::new(FlightSqlDriver::new()))
        .open_table(endpoint, options)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("handshake rejected"));
}
