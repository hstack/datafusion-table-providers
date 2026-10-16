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
use crate::{to_df_err, DefaultFlightChannelProvider, FlightChannelProvider};
use datafusion::common::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;

/// Physical extension codec for FlightExec
#[derive(Clone, Debug)]
pub struct FlightPhysicalCodec {
    channel_provider: Arc<dyn FlightChannelProvider>,
}

impl Default for FlightPhysicalCodec {
    fn default() -> Self {
        Self {
            channel_provider: Arc::new(DefaultFlightChannelProvider),
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
        _registry: &TaskContext,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if inputs.is_empty() {
            let config: FlightConfig = serde_json::from_slice(buf).map_err(to_df_err)?;
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
    ) -> datafusion::common::Result<()> {
        if let Some(flight) = node.downcast_ref::<FlightExec>() {
            let mut bytes = serde_json::to_vec(flight.config()).map_err(to_df_err)?;
            buf.append(&mut bytes);
            Ok(())
        } else {
            Err(DataFusionError::Internal(
                "This codec only supports the FlightExec physical plan".into(),
            ))
        }
    }
}
