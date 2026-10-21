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

//! Configured metadata access shared by Flight table providers.

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use arrow_flight::flight_service_client::FlightServiceClient;
use datafusion::common::{DataFusionError, Result};
use tonic::transport::Channel;

use crate::{
    flight_service_client, to_df_err, FlightChannelProvider, FlightDriver, FlightFactoryOptions,
    FlightMetadata,
};

pub(crate) async fn connect_and_fetch_metadata(
    origin: &str,
    driver: &dyn FlightDriver,
    factory_options: &FlightFactoryOptions,
    channel_provider: &dyn FlightChannelProvider,
    options: &HashMap<String, String>,
) -> Result<(FlightServiceClient<Channel>, FlightMetadata)> {
    with_metadata_timeout(origin, factory_options.metadata_timeout, async {
        let channel = channel_provider.channel(origin).await?;
        let client = flight_service_client(channel, &factory_options.size_limits);
        let metadata = driver
            .metadata(client.clone(), factory_options.properties.clone(), options)
            .await
            .map_err(to_df_err)?;
        Ok((client, metadata))
    })
    .await
}

pub(crate) async fn fetch_metadata(
    origin: &str,
    client: FlightServiceClient<Channel>,
    driver: &dyn FlightDriver,
    factory_options: &FlightFactoryOptions,
    options: &HashMap<String, String>,
) -> Result<FlightMetadata> {
    with_metadata_timeout(origin, factory_options.metadata_timeout, async {
        driver
            .metadata(client, factory_options.properties.clone(), options)
            .await
            .map_err(to_df_err)
    })
    .await
}

async fn with_metadata_timeout<T>(
    origin: &str,
    timeout: Option<Duration>,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, future).await.map_err(|_| {
            DataFusionError::Execution(format!(
                "Flight metadata operation for '{origin}' timed out after {timeout:?}"
            ))
        })?,
        None => future.await,
    }
}
