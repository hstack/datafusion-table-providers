// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Generic [FlightTableFactory] that can connect to Arrow Flight services,
//! with a [sql::FlightSqlDriver] provided out-of-the-box.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use crate::access::{connect_and_fetch_metadata, fetch_metadata};
use crate::exec::FlightExec;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::FlightInfo;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::common::stats::Precision;
use datafusion::common::{project_schema, DataFusionError, Statistics};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::{CreateExternalTable, Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use serde::{Deserialize, Serialize};
use tonic::transport::Channel;

mod access;
mod channel;
pub mod codec;
mod exec;
pub mod sql;

pub use channel::{DefaultFlightChannelProvider, FlightChannelProvider};
pub use exec::enforce_schema;

/// Generic Arrow Flight data source. Requires a [FlightDriver] that allows implementors
/// to integrate any custom Flight RPC service by producing a [FlightMetadata] for some DDL.
///
/// # Sample usage:
/// ```
/// # use arrow_flight::{FlightClient, FlightDescriptor};
/// # use arrow_flight::flight_service_client::FlightServiceClient;
/// # use datafusion::prelude::SessionContext;
/// # use datafusion_table_providers_flightsql::{FlightDriver, FlightFactoryOptions, FlightMetadata, FlightProperties, FlightTableFactory};
/// # use std::collections::HashMap;
/// # use std::sync::Arc;
/// # use std::time::Duration;
/// # use tonic::transport::Channel;
///
/// #[derive(Debug, Clone, Default)]
/// struct CustomFlightDriver {}
///
/// #[async_trait::async_trait]
/// impl FlightDriver for CustomFlightDriver {
///     async fn metadata(
///         &self,
///         client: FlightServiceClient<Channel>,
///         properties: FlightProperties,
///         opts: &HashMap<String, String>,
///     ) -> arrow_flight::error::Result<FlightMetadata> {
///         let mut client = FlightClient::new_from_inner(client);
///         // for simplicity, we'll just assume the server expects a string command and no handshake
///         let descriptor = FlightDescriptor::new_cmd(opts["flight.command"].clone());
///         let flight_info = client.get_flight_info(descriptor).await?;
///         FlightMetadata::try_new(flight_info, properties)
///     }
/// }
///
/// #[tokio::main]
/// async fn main() {
///     let ctx = SessionContext::new();
///     let factory = FlightTableFactory::new(Arc::new(CustomFlightDriver::default()))
///         .with_options(
///             FlightFactoryOptions::new().with_metadata_timeout(Duration::from_secs(30)),
///         );
///     ctx.state_ref().write().table_factories_mut().insert(
///         "CUSTOM_FLIGHT".into(),
///         Arc::new(factory),
///     );
///     let _ = ctx.sql(
///         r#"
///         CREATE EXTERNAL TABLE custom_flight_table STORED AS CUSTOM_FLIGHT
///         LOCATION 'https://custom.flight.rpc'
///         OPTIONS ('flight.command' 'AI, show me the data!')
///     "#,
///     ); // no .await here, so we don't actually try to connect to the bogus URL
/// }
/// ```
#[derive(Clone, Debug)]
pub struct FlightTableFactory {
    driver: Arc<dyn FlightDriver>,
    channel_provider: Arc<dyn FlightChannelProvider>,
    options: FlightFactoryOptions,
}

impl FlightTableFactory {
    /// Create a data source using the provided driver
    pub fn new(driver: Arc<dyn FlightDriver>) -> Self {
        Self {
            driver,
            channel_provider: Arc::new(DefaultFlightChannelProvider),
            options: FlightFactoryOptions::default(),
        }
    }

    /// Configure request policy and the default properties passed to the driver.
    pub fn with_options(mut self, options: FlightFactoryOptions) -> Self {
        self.options = options;
        self
    }

    /// Supply channels for metadata and execution RPCs.
    /// If you decide to supply a custom channel provider, you likely also need
    /// to configure the codec with [`codec::FlightPhysicalCodec::with_channel_provider`].
    pub fn with_channel_provider(
        mut self,
        channel_provider: Arc<dyn FlightChannelProvider>,
    ) -> Self {
        self.channel_provider = channel_provider;
        self
    }

    /// Convenient way to create a [FlightTable] programatically, as an alternative to DDL.
    pub async fn open_table(
        &self,
        entry_point: impl Into<String>,
        options: HashMap<String, String>,
    ) -> datafusion::common::Result<FlightTable> {
        let origin = entry_point.into();
        let (client, metadata) = connect_and_fetch_metadata(
            &origin,
            self.driver.as_ref(),
            &self.options,
            self.channel_provider.as_ref(),
            &options,
        )
        .await?;

        let num_rows = precision(metadata.info.total_records);
        let total_byte_size = precision(metadata.info.total_bytes);
        let logical_schema = metadata.schema.clone();
        let stats = Statistics {
            num_rows,
            total_byte_size,
            column_statistics: vec![],
        };

        let metadata_supplier = if metadata.props.reusable_flight_info {
            MetadataSupplier::Reusable(Arc::new(metadata))
        } else {
            MetadataSupplier::Refresh {
                driver: self.driver.clone(),
                client: Box::new(client),
                options,
                factory_options: self.options.clone(),
            }
        };

        Ok(FlightTable {
            metadata_supplier,
            origin,
            logical_schema,
            stats,
            channel_provider: self.channel_provider.clone(),
            size_limits: self.options.size_limits,
        })
    }
}

#[async_trait]
impl TableProviderFactory for FlightTableFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> datafusion::common::Result<Arc<dyn TableProvider>> {
        let table = self.open_table(&cmd.location, cmd.options.clone()).await?;
        Ok(Arc::new(table))
    }
}

/// Caller-selected request policy and default table properties.
///
/// Configure these on [`FlightTableFactory::with_options`]. Drivers receive a
/// client with the selected message limits and a copy of the property defaults.
/// Only execution settings, not metadata deadlines, are serialized into plans.
#[derive(Clone, Debug, Default)]
pub struct FlightFactoryOptions {
    size_limits: SizeLimits,
    metadata_timeout: Option<Duration>,
    properties: FlightProperties,
}

impl FlightFactoryOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Max sizes in bytes for encoded/decoded gRPC messages.
    ///
    /// Applies to the metadata client supplied to the driver and to `DoGet`.
    /// Both sizes are unlimited by default.
    pub fn with_size_limits(mut self, size_limits: SizeLimits) -> Self {
        self.size_limits = size_limits;
        self
    }

    /// Limit the time spent acquiring metadata, including channel acquisition.
    ///
    /// A single budget covers the initial channel acquisition and all driver
    /// metadata RPCs, including authentication. Each metadata refresh gets a
    /// fresh budget on the retained channel. By default there is no timeout.
    ///
    /// This does not limit `DoGet` execution or close shared channels on expiry.
    pub fn with_metadata_timeout(mut self, timeout: Duration) -> Self {
        self.metadata_timeout = Some(timeout);
        self
    }

    /// Default table properties supplied to the driver for each metadata call.
    ///
    /// The driver may refine these defaults based on the service's response.
    /// Returned properties govern table scans and execution, not request policy.
    pub fn with_properties(mut self, properties: FlightProperties) -> Self {
        self.properties = properties;
        self
    }
}

/// Extension point for integrating any Flight RPC service as a [FlightTableFactory].
/// Handles the initial `GetFlightInfo` call and all its prerequisites (such as `Handshake`),
/// to produce a [FlightMetadata].
#[async_trait]
pub trait FlightDriver: Sync + Send + Debug {
    /// Returns a [FlightMetadata] according to the provided table options.
    ///
    /// Use the supplied client when constructing an Arrow Flight client.
    /// Preserve or refine the supplied default properties in the returned metadata.
    /// The driver must provide at least a [FlightInfo] in order to construct a flight metadata.
    async fn metadata(
        &self,
        client: FlightServiceClient<Channel>,
        properties: FlightProperties,
        options: &HashMap<String, String>,
    ) -> arrow_flight::error::Result<FlightMetadata>;
}

/// The information that a [FlightDriver] must produce
/// in order to register flights as DataFusion tables.
#[derive(Clone, Debug)]
pub struct FlightMetadata {
    /// FlightInfo object produced by the driver
    info: FlightInfo,
    /// Driver-resolved table semantics and execution headers
    props: FlightProperties,
    /// Arrow schema. Can be enforced by the driver or inferred from the FlightInfo
    schema: SchemaRef,
}

impl FlightMetadata {
    /// Customize everything that is in the driver's control
    pub fn new(info: FlightInfo, props: FlightProperties, schema: SchemaRef) -> Self {
        Self {
            info,
            props,
            schema,
        }
    }

    /// Customize flight properties and try to use the FlightInfo schema
    pub fn try_new(info: FlightInfo, props: FlightProperties) -> arrow_flight::error::Result<Self> {
        let schema = Arc::new(info.clone().try_decode_schema()?);
        Ok(Self::new(info, props, schema))
    }
}

impl TryFrom<FlightInfo> for FlightMetadata {
    type Error = FlightError;

    fn try_from(info: FlightInfo) -> Result<Self, Self::Error> {
        Self::try_new(info, FlightProperties::default())
    }
}

/// Table semantics and execution headers resolved by a [`FlightDriver`].
///
/// Also used as driver input defaults through [`FlightFactoryOptions::with_properties`].
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct FlightProperties {
    unbounded_streams: bool,
    grpc_headers: HashMap<String, String>,
    reusable_flight_info: bool,
}

impl FlightProperties {
    pub fn new() -> Self {
        Default::default()
    }

    /// Whether the service will produce infinite streams
    pub fn with_unbounded_streams(mut self, unbounded_streams: bool) -> Self {
        self.unbounded_streams = unbounded_streams;
        self
    }

    /// Whether the service will produce infinite streams.
    pub fn unbounded_streams(&self) -> bool {
        self.unbounded_streams
    }

    /// gRPC headers to use on subsequent calls.
    pub fn with_grpc_headers(mut self, grpc_headers: HashMap<String, String>) -> Self {
        self.grpc_headers = grpc_headers;
        self
    }

    /// gRPC headers to use on subsequent calls.
    pub fn grpc_headers(&self) -> &HashMap<String, String> {
        &self.grpc_headers
    }

    /// Whether the FlightInfo objects produced by the service can be used multiple times
    /// or need to be refreshed before every table scan.
    pub fn with_reusable_flight_info(mut self, reusable_flight_info: bool) -> Self {
        self.reusable_flight_info = reusable_flight_info;
        self
    }

    /// Whether the FlightInfo can be reused without refreshing it before a scan.
    pub fn reusable_flight_info(&self) -> bool {
        self.reusable_flight_info
    }
}

/// Message size limits to be passed to the underlying gRPC library.
#[derive(Copy, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SizeLimits {
    encoding: usize,
    decoding: usize,
}

impl SizeLimits {
    pub fn new(encoding: usize, decoding: usize) -> Self {
        Self { encoding, decoding }
    }
}

impl Default for SizeLimits {
    fn default() -> Self {
        Self {
            // no limits
            encoding: usize::MAX,
            decoding: usize::MAX,
        }
    }
}

#[derive(Clone, Debug)]
enum MetadataSupplier {
    Reusable(Arc<FlightMetadata>),
    Refresh {
        driver: Arc<dyn FlightDriver>,
        client: Box<FlightServiceClient<Channel>>,
        options: HashMap<String, String>,
        factory_options: FlightFactoryOptions,
    },
}

impl MetadataSupplier {
    async fn flight_metadata(
        &self,
        origin: &str,
    ) -> datafusion::common::Result<Arc<FlightMetadata>> {
        match self {
            Self::Reusable(metadata) => Ok(metadata.clone()),
            Self::Refresh {
                driver,
                client,
                options,
                factory_options,
            } => fetch_metadata(
                origin,
                client.as_ref().clone(),
                driver.as_ref(),
                factory_options,
                options,
            )
            .await
            .map(Arc::new),
        }
    }
}

/// Table provider that wraps a specific flight from an Arrow Flight service
pub struct FlightTable {
    metadata_supplier: MetadataSupplier,
    origin: String,
    logical_schema: SchemaRef,
    stats: Statistics,
    channel_provider: Arc<dyn FlightChannelProvider>,
    size_limits: SizeLimits,
}

impl std::fmt::Debug for FlightTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlightTable")
            .field("origin", &self.origin)
            .field("logical_schema", &self.logical_schema)
            .field("stats", &self.stats)
            .finish()
    }
}

#[async_trait]
impl TableProvider for FlightTable {
    fn schema(&self) -> SchemaRef {
        self.logical_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let metadata = self.metadata_supplier.flight_metadata(&self.origin).await?;
        let output_schema = project_schema(&metadata.schema, projection)?;
        Ok(Arc::new(FlightExec::try_new(
            metadata.as_ref(),
            output_schema,
            &self.origin,
            self.size_limits,
            self.channel_provider.clone(),
        )?))
    }

    fn statistics(&self) -> Option<Statistics> {
        Some(self.stats.clone())
    }
}

fn to_df_err<E: Error + Send + Sync + 'static>(err: E) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}

fn flight_service_client(
    channel: Channel,
    size_limits: &SizeLimits,
) -> FlightServiceClient<Channel> {
    FlightServiceClient::new(channel)
        .max_encoding_message_size(size_limits.encoding)
        .max_decoding_message_size(size_limits.decoding)
}

fn precision(total: i64) -> Precision<usize> {
    if total < 0 {
        Precision::Absent
    } else {
        Precision::Exact(total as usize)
    }
}
