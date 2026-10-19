use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::FlightInfo;
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::common::{DataFusionError, Result};
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::prelude::SessionContext;
use datafusion_table_providers::flight::sql::{FlightSqlDriver, HEADER_PREFIX};
use datafusion_table_providers::flight::{
    FlightChannelProvider, FlightDriver, FlightFactoryOptions, FlightMetadata, FlightProperties,
    FlightTableFactory,
};
use rstest::rstest;
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
