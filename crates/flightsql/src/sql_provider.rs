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

//! SQL-specific Flight sources and table options.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_flight::flight_service_client::FlightServiceClient;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{project_schema, DataFusionError, Result, Statistics};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use tonic::transport::Channel;

use crate::access::{connect_and_fetch_metadata, fetch_metadata};
use crate::exec::FlightExec;
use crate::sql::{FlightSqlDriver, QUERY};
use crate::sql_gen::{classify_filter, discovery_sql, normalize_query, scan_sql};
use crate::{DefaultFlightChannelProvider, FlightChannelProvider, FlightFactoryOptions};

/// A literal remote table identifier or a query used as a derived relation.
///
/// Constructors validate and normalize the source without resolving remote
/// tables or functions locally. Direct enum variants must be normalized again
/// before opening a table.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum FlightSqlSource {
    /// A single literal identifier, including any dots or embedded quotes.
    Table(String),
    /// One query parsed with the generic SQL dialect.
    Query(String),
}

impl FlightSqlSource {
    /// Trim and validate a literal table identifier without splitting dotted names.
    pub fn table(name: impl Into<String>) -> Result<Self> {
        Self::Table(name.into()).normalized()
    }

    /// Parse and normalize one query, removing a final statement terminator.
    ///
    /// Syntax validation is not a privilege sandbox. Remote authorization still
    /// applies, and functions in a query may have side effects.
    pub fn query(sql: impl Into<String>) -> Result<Self> {
        Self::Query(sql.into()).normalized()
    }

    pub(crate) fn normalized(self) -> Result<Self> {
        match self {
            Self::Table(name) => {
                let name = name.trim();
                if name.is_empty() || name.contains('\0') {
                    return Err(DataFusionError::Plan(
                        "Flight SQL table identifier must be nonempty and contain no NUL bytes"
                            .into(),
                    ));
                }
                Ok(Self::Table(name.to_owned()))
            }
            Self::Query(sql) => Ok(Self::Query(normalize_query(&sql)?)),
        }
    }
}

/// SQL-provider policy, separate from Flight transport and authentication options.
///
/// Predicate pushdown is disabled by default. Selecting a compatibility policy
/// is a caller assertion about the remote service, not capability discovery.
#[derive(Clone, Debug, Default)]
pub struct FlightSqlTableOptions {
    datafusion_compatible_pushdown: bool,
}

impl FlightSqlTableOptions {
    /// Enable renderable table predicates with a DataFusion-compatible function allowlist.
    ///
    /// The remote SQL engine must support the emitted expressions and allowlisted
    /// functions with compatible semantics. This setting does not discover server
    /// capabilities. Filters remain inexact and are rechecked locally; DataFusion
    /// controls safe limit pushdown.
    ///
    /// Allowlisted functions: `abs`, `ceil`, `floor`, `round`, `sqrt`, `exp`,
    /// `ln`, `power`, `lower`, `upper`, and `coalesce`.
    ///
    /// Query-source outer filters remain local regardless of this setting.
    pub fn with_datafusion_compatible_pushdown(mut self) -> Self {
        self.datafusion_compatible_pushdown = true;
        self
    }

    pub(crate) fn pushdown_enabled(&self) -> bool {
        self.datafusion_compatible_pushdown
    }
}

/// Opens SQL sources using full-schema discovery and fresh tickets for each scan.
///
/// Discovery and projection require DataFusion-compatible SQL syntax. No remote
/// dialect or predicate capabilities are inferred from successful discovery.
#[derive(Clone, Debug)]
pub struct FlightSqlTableFactory {
    driver: FlightSqlDriver,
    options: FlightFactoryOptions,
    channel_provider: Arc<dyn FlightChannelProvider>,
    table_options: FlightSqlTableOptions,
}

impl FlightSqlTableFactory {
    pub fn new(driver: FlightSqlDriver) -> Self {
        Self {
            driver,
            options: FlightFactoryOptions::default(),
            channel_provider: Arc::new(DefaultFlightChannelProvider),
            table_options: FlightSqlTableOptions::default(),
        }
    }

    /// Configure transport, metadata deadlines, and default execution properties.
    pub fn with_options(mut self, options: FlightFactoryOptions) -> Self {
        self.options = options;
        self
    }

    /// Supply channels for metadata and execution RPCs.
    ///
    /// Distributed workers must configure the physical codec's channel provider too.
    pub fn with_channel_provider(
        mut self,
        channel_provider: Arc<dyn FlightChannelProvider>,
    ) -> Self {
        self.channel_provider = channel_provider;
        self
    }

    /// Select SQL-specific policy separately from transport/authentication options.
    pub fn with_table_options(mut self, table_options: FlightSqlTableOptions) -> Self {
        self.table_options = table_options;
        self
    }

    /// Discover the canonical schema without fetching or retaining discovery tickets.
    ///
    /// The factory owns generated SQL; a caller-supplied [`QUERY`] option is an error.
    pub async fn open_table(
        &self,
        endpoint: impl Into<String>,
        source: FlightSqlSource,
        options: HashMap<String, String>,
    ) -> Result<FlightSqlTable> {
        let source = source.normalized()?;
        if options.contains_key(QUERY) {
            return Err(DataFusionError::Plan(
                "Flight SQL source conflicts with caller-supplied flight.sql.query".into(),
            ));
        }

        let origin = endpoint.into();
        let mut discovery_options = options.clone();
        discovery_options.insert(QUERY.into(), discovery_sql(&source));
        let (client, metadata) = connect_and_fetch_metadata(
            &origin,
            &self.driver,
            &self.options,
            self.channel_provider.as_ref(),
            &discovery_options,
        )
        .await?;

        let logical_schema = metadata.schema;
        if logical_schema.fields().is_empty() {
            return Err(DataFusionError::Execution(
                "Flight SQL discovery returned an empty schema".into(),
            ));
        }
        let mut names = HashSet::new();
        for field in logical_schema.fields() {
            if !names.insert(field.name()) {
                return Err(DataFusionError::Execution(format!(
                    "Flight SQL discovery schema has ambiguous duplicate field {:?}",
                    field.name()
                )));
            }
        }

        Ok(FlightSqlTable {
            source,
            logical_schema,
            options,
            table_options: self.table_options.clone(),
            client,
            driver: self.driver.clone(),
            origin,
            factory_options: self.options.clone(),
            channel_provider: self.channel_provider.clone(),
        })
    }
}

/// A query-local SQL source with an immutable, unprojected discovery schema.
///
/// Each scan submits new SQL, validates its metadata schema, and creates a plan
/// containing only that submission's tickets and execution properties.
/// Required fields must retain their types and must not weaken non-null
/// guarantees. Schema-level and top-level field custom metadata may differ.
pub struct FlightSqlTable {
    source: FlightSqlSource,
    logical_schema: SchemaRef,
    options: HashMap<String, String>,
    table_options: FlightSqlTableOptions,
    client: FlightServiceClient<Channel>,
    driver: FlightSqlDriver,
    origin: String,
    factory_options: FlightFactoryOptions,
    channel_provider: Arc<dyn FlightChannelProvider>,
}

impl fmt::Debug for FlightSqlTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlightSqlTable")
            .field("source", &self.source)
            .field("logical_schema", &self.logical_schema)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for FlightSqlTable {
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
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let query = scan_sql(
            &self.source,
            &self.logical_schema,
            projection.map(Vec::as_slice),
            filters,
            limit,
        )?;
        let mut options = self.options.clone();
        options.insert(QUERY.into(), query);
        let metadata = fetch_metadata(
            &self.origin,
            self.client.clone(),
            &self.driver,
            &self.factory_options,
            &options,
        )
        .await?;

        let output_schema = project_schema(&self.logical_schema, projection)?;
        validate_scan_schema(&output_schema, &metadata.schema)?;

        Ok(Arc::new(FlightExec::try_new(
            &metadata,
            output_schema,
            &self.origin,
            self.factory_options.size_limits,
            self.channel_provider.clone(),
        )?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|filter| classify_filter(&self.source, &self.table_options, filter))
            .collect()
    }

    fn statistics(&self) -> Option<Statistics> {
        None
    }
}

/// Ignore schema-level and top-level field custom metadata only. Nested field
/// definitions remain part of the exact data type contract.
fn validate_scan_schema(expected: &SchemaRef, actual: &SchemaRef) -> Result<()> {
    for expected in expected.fields() {
        let actual = actual.field_with_name(expected.name()).map_err(|_| {
            DataFusionError::Execution(format!(
                "Flight SQL scan schema is missing field {:?}",
                expected.name()
            ))
        })?;
        if expected.data_type() != actual.data_type()
            || (!expected.is_nullable() && actual.is_nullable())
        {
            return Err(DataFusionError::Execution(format!(
                "Flight SQL scan schema changed for field {:?}: expected {:?} nullable={}, received {:?} nullable={}",
                expected.name(), expected.data_type(), expected.is_nullable(),
                actual.data_type(), actual.is_nullable()
            )));
        }
    }
    Ok(())
}
