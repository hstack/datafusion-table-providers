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

//! Execution plan for reading flights from Arrow Flight services

use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::str::FromStr;
use std::sync::Arc;

use crate::{
    flight_service_client, to_df_err, FlightChannelProvider, FlightMetadata, FlightProperties,
    SizeLimits,
};
use arrow_flight::error::FlightError;
use arrow_flight::{FlightClient, FlightEndpoint, Ticket};
use datafusion::arrow::array::{new_null_array, ArrayRef, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::ToByteSlice;
use datafusion::arrow::datatypes::{Field, SchemaRef};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::record_batch::RecordBatchOptions;
use datafusion::common::DataFusionError;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_table_providers_common::sql::db_connection_pool::runtime::run_async_with_tokio;
use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use tonic::metadata::{AsciiMetadataKey, MetadataMap};

/// Arrow Flight physical plan that maps flight endpoints to partitions
#[derive(Clone, Debug)]
pub(crate) struct FlightExec {
    config: FlightConfig,
    channel_provider: Arc<dyn FlightChannelProvider>,
    plan_properties: Arc<PlanProperties>,
    metadata_map: Arc<MetadataMap>,
}

impl FlightExec {
    /// Creates a FlightExec with an explicit output schema and the provided [FlightMetadata].
    /// Uses the origin URL as the fallback location as per the protocol spec.
    pub(crate) fn try_new(
        metadata: &FlightMetadata,
        output_schema: SchemaRef,
        origin: &str,
        size_limits: SizeLimits,
        channel_provider: Arc<dyn FlightChannelProvider>,
    ) -> Result<Self> {
        let partitions = metadata
            .info
            .endpoint
            .iter()
            .map(|endpoint| FlightPartition::new(endpoint, origin.to_string()))
            .collect();
        let config = FlightConfig {
            origin: origin.into(),
            schema: output_schema,
            partitions,
            properties: metadata.props.clone(),
            size_limits,
        };
        Ok(Self::from_config(config, channel_provider))
    }

    pub(crate) fn config(&self) -> &FlightConfig {
        &self.config
    }

    pub(crate) fn from_config(
        config: FlightConfig,
        channel_provider: Arc<dyn FlightChannelProvider>,
    ) -> Self {
        let exec_mode = if config.properties.unbounded_streams {
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            }
        } else {
            Boundedness::Bounded
        };
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(config.schema.clone()),
            Partitioning::UnknownPartitioning(config.partitions.len()),
            EmissionType::Incremental,
            exec_mode,
        ));
        let mut mm = MetadataMap::new();
        for (k, v) in config.properties.grpc_headers.iter() {
            let key = AsciiMetadataKey::from_str(k.as_str()).expect("invalid header name");
            let value = v.parse().expect("invalid header value");
            mm.insert(key, value);
        }
        Self {
            config,
            channel_provider,
            plan_properties,
            metadata_map: Arc::from(mm),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct FlightConfig {
    origin: String,
    schema: SchemaRef,
    partitions: Arc<[FlightPartition]>,
    properties: FlightProperties,
    size_limits: SizeLimits,
}

/// The minimum information required for fetching a flight stream.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FlightPartition {
    locations: Arc<[String]>,
    ticket: FlightTicket,
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
struct FlightTicket(Arc<[u8]>);

impl From<Option<&Ticket>> for FlightTicket {
    fn from(ticket: Option<&Ticket>) -> Self {
        let bytes = match ticket {
            Some(t) => t.ticket.to_byte_slice().into(),
            None => [].into(),
        };
        Self(bytes)
    }
}

impl Debug for FlightTicket {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[..{} bytes..]", self.0.len())
    }
}

impl FlightPartition {
    fn new(endpoint: &FlightEndpoint, fallback_location: String) -> Self {
        let locations = if endpoint.location.is_empty() {
            [fallback_location].into()
        } else {
            endpoint
                .location
                .iter()
                .map(|loc| {
                    if loc.uri.starts_with("arrow-flight-reuse-connection://") {
                        fallback_location.clone()
                    } else {
                        loc.uri.clone()
                    }
                })
                .collect()
        };

        Self {
            locations,
            ticket: endpoint.ticket.as_ref().into(),
        }
    }
}

async fn flight_client(
    source: &str,
    grpc_headers: &MetadataMap,
    size_limits: &SizeLimits,
    channel_provider: &dyn FlightChannelProvider,
) -> Result<FlightClient> {
    let channel = channel_provider.channel(source).await?;
    let inner_client = flight_service_client(channel, size_limits);
    let mut client = FlightClient::new_from_inner(inner_client);
    client.metadata_mut().clone_from(grpc_headers);
    Ok(client)
}

async fn flight_stream(
    partition: FlightPartition,
    schema: SchemaRef,
    grpc_headers: Arc<MetadataMap>,
    size_limits: SizeLimits,
    channel_provider: Arc<dyn FlightChannelProvider>,
) -> Result<SendableRecordBatchStream> {
    let mut errors: Vec<Box<dyn Error + Send + Sync>> = vec![];
    for loc in partition.locations.iter() {
        let get_client = || async {
            flight_client(
                loc,
                grpc_headers.as_ref(),
                &size_limits,
                channel_provider.as_ref(),
            )
            .await
        };
        let client = run_async_with_tokio(get_client).await?;
        match try_fetch_stream(client, &partition.ticket, schema.clone()).await {
            Ok(stream) => return Ok(stream),
            Err(e) => errors.push(Box::new(e)),
        }
    }
    let err = errors.into_iter().next_back().unwrap_or_else(|| {
        Box::new(FlightError::ProtocolError(format!(
            "No available location for endpoint {:?}",
            partition.locations
        )))
    });
    Err(DataFusionError::External(err))
}

async fn try_fetch_stream(
    mut client: FlightClient,
    ticket: &FlightTicket,
    schema: SchemaRef,
) -> arrow_flight::error::Result<SendableRecordBatchStream> {
    let ticket = Ticket::new(ticket.0.to_vec());
    let stream = client.do_get(ticket).await?.map_err(to_df_err);
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream.map(move |item| item.and_then(|rb| enforce_schema(rb, &schema).map_err(Into::into))),
    )))
}

/// Best-effort morphing of a record batch into a new schema.
/// - drops columns if the corresponding field is missing from the target schema
/// - preserves row counts in zero-column batches when the target schema is empty
/// - attempts a column-wise cast if data types are different
/// - generates missing columns as all-nulls if the corresponding field is nullable
pub fn enforce_schema(
    batch: RecordBatch,
    target_schema: &SchemaRef,
) -> datafusion::arrow::error::Result<RecordBatch> {
    if batch.schema() == *target_schema {
        Ok(batch)
    } else if target_schema.fields.is_empty() {
        RecordBatch::try_new_with_options(
            target_schema.to_owned(),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
    } else if target_schema.contains(batch.schema_ref()) {
        batch.with_schema(target_schema.to_owned())
    } else {
        let columns = target_schema
            .fields
            .iter()
            .map(|field| find_matching_column(&batch, field.as_ref()))
            .collect::<datafusion::arrow::error::Result<_>>()?;
        RecordBatch::try_new(target_schema.to_owned(), columns)
    }
}

/// For a target schema field, extract the column with the same name if present in the
/// record batch and cast it to the desired data type if needed. If the column is missing
/// but the target schema field is nullable, generates a null-array column.
fn find_matching_column(
    batch: &RecordBatch,
    field: &Field,
) -> datafusion::arrow::error::Result<ArrayRef> {
    if let Some(column) = batch.column_by_name(field.name()) {
        if column.data_type() == field.data_type() {
            Ok(column.to_owned())
        } else {
            cast(column.as_ref(), field.data_type())
        }
    } else if field.is_nullable() {
        Ok(new_null_array(field.data_type(), batch.num_rows()))
    } else {
        Err(ArrowError::SchemaError(format!(
            "Required field `{}` is missing from the original record batch",
            field.name()
        )))
    }
}

impl DisplayAs for FlightExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::TreeRender => write!(
                f,
                "FlightExec: origin={}, streams={}",
                self.config.origin,
                self.config.partitions.len()
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "FlightExec: origin={}, partitions={:?}, properties={:?}",
                self.config.origin, self.config.partitions, self.config.properties,
            ),
        }
    }
}

impl ExecutionPlan for FlightExec {
    fn name(&self) -> &str {
        "FlightExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let future_stream = flight_stream(
            self.config.partitions[partition].clone(),
            self.schema(),
            self.metadata_map.clone(),
            self.config.size_limits,
            self.channel_provider.clone(),
        );
        let stream = futures::stream::once(future_stream).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::exec::{enforce_schema, FlightConfig, FlightExec, FlightPartition, FlightTicket};
    use crate::{DefaultFlightChannelProvider, FlightMetadata, FlightProperties, SizeLimits};
    use arrow_flight::FlightInfo;
    use datafusion::arrow::array::{
        BooleanArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
    use datafusion::physical_plan::ExecutionPlan;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_flight_config_serde() {
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("f1", DataType::Utf8, true)),
            Arc::new(Field::new("f2", DataType::Int32, false)),
        ]));
        let partitions = [
            FlightPartition {
                locations: ["l1".into(), "l2".into()].into(),
                ticket: FlightTicket("ticket1".as_bytes().into()),
            },
            FlightPartition {
                locations: ["l3".into(), "l4".into()].into(),
                ticket: FlightTicket("ticket2".as_bytes().into()),
            },
        ]
        .into();
        let properties = FlightProperties::default()
            .with_unbounded_streams(true)
            .with_grpc_headers(HashMap::from([
                ("h1".into(), "v1".into()),
                ("h2".into(), "v2".into()),
            ]));
        let config = FlightConfig {
            origin: "http://localhost:50050".into(),
            schema,
            partitions,
            properties,
            size_limits: SizeLimits::new(1024, 1024),
        };
        let json = serde_json::to_vec(&config).expect("cannot encode config as json");
        let restored = serde_json::from_slice(json.as_slice()).expect("cannot decode json config");
        assert_eq!(config, restored);
    }

    #[test]
    fn zero_column_adaptation_preserves_rows() {
        let placeholder_schema =
            Arc::new(Schema::new(vec![Field::new("1", DataType::Int64, false)]));
        let output_schema = Arc::new(
            Schema::empty().with_metadata(HashMap::from([("source".into(), "fixture".into())])),
        );
        let mut total_rows = 0;
        for rows in [3, 0, 2] {
            let batch = RecordBatch::try_new(
                placeholder_schema.clone(),
                vec![Arc::new(Int64Array::from(vec![1; rows]))],
            )
            .unwrap();
            let adapted = enforce_schema(batch, &output_schema).unwrap();
            assert_eq!(adapted.num_columns(), 0);
            assert_eq!(adapted.num_rows(), rows);
            assert_eq!(adapted.schema(), output_schema);
            total_rows += adapted.num_rows();
        }
        assert_eq!(total_rows, 5);
    }

    #[test]
    fn explicit_schema_does_not_reapply_projection() {
        let canonical_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Int64, false),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("c", DataType::Int64, false),
            Field::new("a", DataType::Int32, false),
        ]));
        let remote_batch = RecordBatch::try_new(
            output_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![30, 31])),
                Arc::new(Int32Array::from(vec![10, 11])),
            ],
        )
        .unwrap();
        let info = FlightInfo::new()
            .try_with_schema(output_schema.as_ref())
            .unwrap();
        let metadata = FlightMetadata::new(info, FlightProperties::new(), canonical_schema);
        let explicit = FlightExec::try_new(
            &metadata,
            output_schema.clone(),
            "http://origin.invalid",
            SizeLimits::default(),
            Arc::new(DefaultFlightChannelProvider),
        )
        .unwrap();
        assert_eq!(explicit.schema(), output_schema);
        assert_eq!(
            enforce_schema(remote_batch.clone(), &explicit.schema()).unwrap(),
            remote_batch
        );
    }

    #[test]
    fn test_schema_enforcement() {
        let data = StructArray::new(
            Fields::from(vec![
                Arc::new(Field::new("f_int", DataType::Int32, true)),
                Arc::new(Field::new("f_bool", DataType::Boolean, false)),
            ]),
            vec![
                Arc::new(Int32Array::from(vec![10, 20])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ],
            None,
        );
        let input_rb = RecordBatch::from(data);

        let coerced_rb = enforce_schema(
            input_rb.clone(),
            &Arc::new(Schema::new(vec![
                // compatible yet different types with flipped nullability
                Arc::new(Field::new("f_int", DataType::Float32, false)),
                Arc::new(Field::new("f_bool", DataType::Utf8, true)),
            ])),
        )
        .expect("error enforcing a compatible schema");
        assert_ne!(input_rb, coerced_rb);
        assert_eq!(coerced_rb.num_columns(), 2);
        assert_eq!(coerced_rb.num_rows(), 2);
        assert_eq!(
            coerced_rb.column(0).as_ref(),
            &Float32Array::from(vec![10.0, 20.0])
        );
        assert_eq!(
            coerced_rb.column(1).as_ref(),
            &StringArray::from(vec!["true", "false"])
        );

        let projection_rb = enforce_schema(
            input_rb.clone(),
            &Arc::new(Schema::new(vec![
                // keep only the first column and make it non-nullable int16
                Arc::new(Field::new("f_int", DataType::Int16, false)),
            ])),
        )
        .expect("error enforcing a compatible subset of fields");
        assert_eq!(projection_rb.num_columns(), 1);
        assert_eq!(projection_rb.num_rows(), 2);
        assert_eq!(projection_rb.schema().fields().len(), 1);
        assert_eq!(projection_rb.schema().fields()[0].name(), "f_int");

        let incompatible_schema_attempt = enforce_schema(
            input_rb.clone(),
            &Arc::new(Schema::new(vec![
                Arc::new(Field::new("f_int", DataType::Float32, true)),
                Arc::new(Field::new("f_bool", DataType::Date32, false)),
            ])),
        )
        .expect_err("got successful result for incompatible schema");
        assert_eq!(
            incompatible_schema_attempt.to_string(),
            "Cast error: Casting from Boolean to Date32 not supported"
        );

        let enhanced_with_null_column = enforce_schema(
            input_rb.clone(),
            &Arc::new(Schema::new(vec![
                Arc::new(Field::new("f_int", DataType::Int32, true)),
                Arc::new(Field::new("f_bool", DataType::Boolean, false)),
                Arc::new(Field::new("f_extra", DataType::Utf8, true)),
            ])),
        )
        .expect("error adding a nullable extra field");
        assert_eq!(enhanced_with_null_column.num_columns(), 3);
        assert_eq!(
            enhanced_with_null_column.column(2).as_ref(),
            &StringArray::new_null(2)
        );

        let non_nullable_extra_field_attempt = enforce_schema(
            input_rb.clone(),
            &Arc::new(Schema::new(vec![
                Arc::new(Field::new("f_int", DataType::Int32, true)),
                Arc::new(Field::new("f_bool", DataType::Boolean, false)),
                Arc::new(Field::new("f_extra", DataType::Utf8, false)),
            ])),
        )
        .expect_err("got successful result for missing non-nullable field");
        assert_eq!(
            non_nullable_extra_field_attempt.to_string(),
            "Schema error: Required field `f_extra` is missing from the original record batch"
        );
    }
}
