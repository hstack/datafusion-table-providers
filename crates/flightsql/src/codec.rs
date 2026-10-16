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

//! Codec to enable running in a distributed environment

use std::sync::Arc;

use crate::exec::{FlightConfig, FlightExec};
use crate::{to_df_err, DefaultFlightChannelProvider, FlightAuthorization, FlightChannelProvider};
use datafusion::common::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::{PhysicalExtensionCodec, PhysicalProtoConverterExtension};

/// Physical extension codec for FlightExec
#[derive(Clone, Debug)]
pub struct FlightPhysicalCodec {
    channel_provider: Arc<dyn FlightChannelProvider>,
}

impl Default for FlightPhysicalCodec {
    fn default() -> Self {
        Self {
            channel_provider: Arc::new(DefaultFlightChannelProvider::default()),
        }
    }
}

impl FlightPhysicalCodec {
    /// Supply channels for decoded execution plans.
    ///
    /// Configure each worker's codec with its local provider. The provider is
    /// runtime-only state and does not change the serialized plan format.
    pub fn with_channel_provider(
        mut self,
        channel_provider: Arc<dyn FlightChannelProvider>,
    ) -> Self {
        self.channel_provider = channel_provider;
        self
    }
}

impl PhysicalExtensionCodec for FlightPhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        registry: &TaskContext,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if inputs.is_empty() {
            let mut config: FlightConfig = serde_json::from_slice(buf).map_err(to_df_err)?;
            let authorization = registry
                .session_config()
                .get_extension::<FlightAuthorization>();
            config
                .properties
                .restore_authorization(authorization.as_deref())?;
            Ok(Arc::new(FlightExec::from_config(
                config,
                self.channel_provider.clone(),
            )))
        } else {
            Err(DataFusionError::Internal(
                "FlightExec is not supposed to have any inputs".into(),
            ))
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> datafusion::common::Result<()> {
        if let Some(flight) = node.downcast_ref::<FlightExec>() {
            let mut config = flight.config().clone();
            config.properties = config.properties.without_authorization();
            let mut bytes = serde_json::to_vec(&config).map_err(to_df_err)?;
            buf.append(&mut bytes);
            Ok(())
        } else {
            Err(DataFusionError::Internal(
                "This codec only supports the FlightExec physical plan".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_flight::{FlightEndpoint, FlightInfo, Location, Ticket};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::prelude::SessionConfig;
    use datafusion_proto::physical_plan::{DefaultPhysicalProtoConverter, PhysicalExtensionCodec};
    use serde_json::json;

    use super::FlightPhysicalCodec;
    use crate::exec::FlightExec;
    use crate::{
        DefaultFlightChannelProvider, FlightAuthorization, FlightMetadata, FlightProperties,
        SizeLimits,
    };

    #[test]
    fn explicit_schema_round_trip_preserves_execution_configuration() {
        let projected_schema = Schema::new(vec![
            Field::new("c", DataType::Int64, true),
            Field::new("a", DataType::Int32, false),
        ]);
        for schema in [projected_schema, Schema::empty()] {
            let output_schema = Arc::new(
                schema.with_metadata(HashMap::from([("source".into(), "fixture".into())])),
            );
            let remote_schema = if output_schema.fields().is_empty() {
                Schema::new(vec![Field::new("1", DataType::Int64, false)])
            } else {
                output_schema.as_ref().clone()
            };
            let info = FlightInfo::new()
                .try_with_schema(&remote_schema)
                .unwrap()
                .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new(vec![0, 1, 2])))
                .with_endpoint(FlightEndpoint {
                    ticket: Some(Ticket::new(vec![3, 4])),
                    location: vec![
                        Location {
                            uri: "arrow-flight-reuse-connection://?".into(),
                        },
                        Location {
                            uri: "http://alternate.invalid".into(),
                        },
                    ],
                    ..Default::default()
                })
                .with_endpoint(FlightEndpoint {
                    ticket: Some(Ticket::new(vec![5, 255])),
                    location: vec![
                        Location {
                            uri: "http://data-a.invalid".into(),
                        },
                        Location {
                            uri: "http://data-b.invalid".into(),
                        },
                    ],
                    ..Default::default()
                });
            let properties = FlightProperties::new()
                .with_reusable_flight_info(true)
                .with_unbounded_streams(true)
                .with_requires_authorization(true)
                .with_grpc_headers(HashMap::from([
                    ("x-fixture".into(), "codec".into()),
                    ("authorization".into(), "secret".into()),
                ]));
            let metadata = FlightMetadata::try_new(info, properties).unwrap();
            let exec = FlightExec::try_new(
                &metadata,
                output_schema.clone(),
                "http://origin.invalid",
                SizeLimits::new(1024, 2048),
                Arc::new(DefaultFlightChannelProvider::default()),
            )
            .unwrap();
            let codec = FlightPhysicalCodec::default();
            let proto_converter = DefaultPhysicalProtoConverter {};
            let mut bytes = vec![];
            codec
                .try_encode(Arc::new(exec), &mut bytes, &proto_converter)
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
                json!({
                    "origin": "http://origin.invalid",
                    "schema": output_schema,
                    "partitions": [
                        {"locations": ["http://origin.invalid"], "ticket": [0, 1, 2]},
                        {
                            "locations": ["http://origin.invalid", "http://alternate.invalid"],
                            "ticket": [3, 4]
                        },
                        {
                            "locations": ["http://data-a.invalid", "http://data-b.invalid"],
                            "ticket": [5, 255]
                        }
                    ],
                    "properties": {
                        "unbounded_streams": true,
                        "grpc_headers": {
                            "x-fixture": "codec"
                        },
                        "reusable_flight_info": true,
                        "requires_authorization": true
                    },
                    "size_limits": {"encoding": 1024, "decoding": 2048}
                })
            );
            assert!(codec
                .try_decode(&bytes, &[], &TaskContext::default(), &proto_converter)
                .is_err());
            let task_ctx = TaskContext::default().with_session_config(
                SessionConfig::new()
                    .with_extension(Arc::new(FlightAuthorization("request-secret".into()))),
            );
            let decoded = codec
                .try_decode(&bytes, &[], &task_ctx, &proto_converter)
                .unwrap();
            let decoded_exec = decoded.downcast_ref::<FlightExec>().unwrap();
            assert_eq!(
                decoded_exec.config().properties.grpc_headers()["authorization"],
                "request-secret"
            );
            assert_eq!(decoded.schema(), output_schema);
            assert_eq!(decoded.output_partitioning().partition_count(), 3);
            let mut reencoded = vec![];
            codec
                .try_encode(decoded, &mut reencoded, &proto_converter)
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&reencoded).unwrap(),
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
            );
        }
    }
}
