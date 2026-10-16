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

//! Channel acquisition for metadata and flight stream requests.

use std::fmt::Debug;

use async_trait::async_trait;
use datafusion::common::Result;
use tonic::transport::{Channel, ClientTlsConfig};

use crate::to_df_err;

/// Supplies channels for Flight RPCs.
///
/// Implementations may cache channels and return cheap clones for concurrent use,
/// or customize connection establishment. Returned channels must be usable on the
/// runtime executing the RPC. Channels do not carry per-request authentication headers.
#[async_trait]
pub trait FlightChannelProvider: Send + Sync + Debug {
    /// Acquire a channel for the origin or an advertised flight endpoint location.
    async fn channel(&self, endpoint: &str) -> Result<Channel>;
}

/// Establishes a new channel on every acquisition, using the enabled TLS roots.
#[derive(Clone, Debug, Default)]
pub struct DefaultFlightChannelProvider;

#[async_trait]
impl FlightChannelProvider for DefaultFlightChannelProvider {
    async fn channel(&self, endpoint: &str) -> Result<Channel> {
        let tls_config = ClientTlsConfig::new().with_enabled_roots();
        Channel::from_shared(endpoint.to_owned())
            .map_err(to_df_err)?
            .tls_config(tls_config)
            .map_err(to_df_err)?
            .connect()
            .await
            .map_err(to_df_err)
    }
}
