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

use datafusion::common::{DataFusionError, Result};

use crate::sql_gen::normalize_query;

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
    /// Select the narrow DataFusion-compatible signed-integer/boolean predicate policy.
    ///
    /// Query-source outer filters remain local regardless of this setting.
    pub fn with_datafusion_compatible_pushdown(mut self) -> Self {
        self.datafusion_compatible_pushdown = true;
        self
    }

    #[allow(dead_code)] // Consumed by the predicate classifier in a later implementation step.
    pub(crate) fn pushdown_enabled(&self) -> bool {
        self.datafusion_compatible_pushdown
    }
}
