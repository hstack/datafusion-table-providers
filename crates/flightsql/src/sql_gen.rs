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

//! SQL source normalization and discovery/scan rendering.

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNode;
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::sql::sqlparser::ast::{Expr as SqlExpr, Statement, Value};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::{Parser, ParserOptions};
use datafusion::sql::unparser::{dialect::Dialect, Unparser};

use crate::{FlightSqlSource, FlightSqlTableOptions};

const FLIGHT_SQL_PUSHABLE_FUNCTIONS: &[&str] = &[
    "abs", "ceil", "floor", "round", "sqrt", "exp", "ln", "power", "lower", "upper", "coalesce",
];

// SQLParser preserves existing escape pairs; pre-escape raw Arrow names and values.
struct FlightSqlDialect;

impl Dialect for FlightSqlDialect {
    fn identifier_quote_style(&self, _identifier: &str) -> Option<char> {
        Some('"')
    }

    fn col_alias_overrides(&self, name: &str) -> Result<Option<String>> {
        Ok(Some(escape_identifier(name)))
    }

    fn string_literal_to_sql(&self, value: &str) -> Option<SqlExpr> {
        Some(SqlExpr::value(Value::SingleQuotedString(
            value.replace('\'', "''"),
        )))
    }
}

pub(crate) fn normalize_query(sql: &str) -> Result<String> {
    // Preserve SQL escaping when rendering the parsed query back to SQL.
    let statements = Parser::new(&GenericDialect {})
        .with_options(ParserOptions::new().with_unescape(false))
        .try_with_sql(sql)
        .and_then(|mut parser| parser.parse_statements())
        .map_err(|error| DataFusionError::Plan(format!("Invalid Flight SQL query: {error}")))?;
    match statements.as_slice() {
        [Statement::Query(query)] => Ok(query.to_string()),
        _ => Err(DataFusionError::Plan(
            "Flight SQL query source must contain exactly one query statement".into(),
        )),
    }
}

/// Render discovery SQL for an already-normalized source.
pub(crate) fn discovery_sql(source: &FlightSqlSource) -> String {
    match source {
        FlightSqlSource::Table(name) => {
            format!("SELECT * FROM {} LIMIT 1", quote_identifier(name))
        }
        FlightSqlSource::Query(query) => {
            format!("SELECT * FROM ({query}) LIMIT 1")
        }
    }
}

/// Render a scan over an already-normalized source and its canonical schema.
pub(crate) fn scan_sql(
    source: &FlightSqlSource,
    schema: &SchemaRef,
    projection: Option<&[usize]>,
    filters: &[Expr],
    limit: Option<usize>,
) -> Result<String> {
    let projection = match projection {
        None => "*".into(),
        Some([]) => "1".into(),
        Some(indices) => indices
            .iter()
            .map(|index| {
                let field = schema.fields().get(*index).ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "Flight SQL projection index {index} is out of bounds for {} fields",
                        schema.fields().len()
                    ))
                })?;
                Ok(quote_identifier(field.name()))
            })
            .collect::<Result<Vec<_>>>()?
            .join(", "),
    };

    let relation = match source {
        FlightSqlSource::Table(name) => quote_identifier(name),
        FlightSqlSource::Query(query) => format!("({query})"),
    };
    let mut sql = format!("SELECT {projection} FROM {relation}");

    if let Some(predicate) = filters.iter().cloned().reduce(Expr::and) {
        let predicate = Unparser::new(&FlightSqlDialect).expr_to_sql(&predicate)?;
        sql.push_str(&format!(" WHERE {predicate}"));
    }

    if let Some(limit) = limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    Ok(sql)
}

pub(crate) fn classify_filter(
    source: &FlightSqlSource,
    table_options: &FlightSqlTableOptions,
    filter: &Expr,
) -> Result<TableProviderFilterPushDown> {
    use TableProviderFilterPushDown::{Inexact, Unsupported};

    match source {
        FlightSqlSource::Query(_) => Ok(Unsupported),
        FlightSqlSource::Table(_) if !table_options.pushdown_enabled() => Ok(Unsupported),
        FlightSqlSource::Table(_) => {
            let unsupported_function = filter.exists(|expr| {
                Ok(matches!(
                    expr,
                    Expr::ScalarFunction(function)
                        if !FLIGHT_SQL_PUSHABLE_FUNCTIONS.iter()
                            .any(|name| name.eq_ignore_ascii_case(function.func.name()))
                ))
            })?;
            if unsupported_function
                || Unparser::new(&FlightSqlDialect)
                    .expr_to_sql(filter)
                    .is_err()
            {
                Ok(Unsupported)
            } else {
                Ok(Inexact)
            }
        }
    }
}

fn quote_identifier(name: &str) -> String {
    format!("\"{}\"", escape_identifier(name))
}

fn escape_identifier(name: &str) -> String {
    name.replace('"', "\"\"")
}

#[cfg(test)]
mod tests {
    use std::ops::Not;
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::common::{Column, DataFusionError};
    use datafusion::functions::math::expr_fn::{abs, sin};
    use datafusion::logical_expr::{expr::Cast, Expr, TableProviderFilterPushDown};
    use datafusion::prelude::{col, lit};

    use super::{classify_filter, discovery_sql, scan_sql};
    use crate::{FlightSqlSource, FlightSqlTableOptions};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, true),
            Field::new("c", DataType::Int64, false),
        ]))
    }

    #[test]
    fn table_discovery_quotes_literal_identifiers() {
        for (input, normalized, expected) in [
            (" a.b ", "a.b", "SELECT * FROM \"a.b\" LIMIT 1"),
            ("select", "select", "SELECT * FROM \"select\" LIMIT 1"),
            (
                "\"order\"",
                "\"order\"",
                "SELECT * FROM \"\"\"order\"\"\" LIMIT 1",
            ),
            ("a\"b", "a\"b", "SELECT * FROM \"a\"\"b\" LIMIT 1"),
            ("a\"\"b", "a\"\"b", "SELECT * FROM \"a\"\"\"\"b\" LIMIT 1"),
            (
                "caf\u{e9}",
                "caf\u{e9}",
                "SELECT * FROM \"caf\u{e9}\" LIMIT 1",
            ),
        ] {
            let source = FlightSqlSource::table(input).unwrap();
            assert_eq!(source, FlightSqlSource::Table(normalized.into()));
            assert_eq!(discovery_sql(&source), expected);
            assert_eq!(source.clone().normalized().unwrap(), source);
        }
    }

    #[test]
    fn invalid_table_sources_are_rejected() {
        for input in ["", " \t\n ", "a\0b"] {
            assert!(matches!(
                FlightSqlSource::table(input),
                Err(DataFusionError::Plan(_))
            ));
            assert!(FlightSqlSource::Table(input.into()).normalized().is_err());
        }
    }

    #[test]
    fn queries_normalize_ast_without_splitting_literals_or_comments() {
        for (input, expected) in [
            (" select 1 as id; ", "SELECT 1 AS id"),
            (
                "SELECT ';' AS marker; -- ignored ; SELECT 2",
                "SELECT ';' AS marker",
            ),
            ("SELECT 1 /* ; SELECT 2 */ AS id;", "SELECT 1 AS id"),
            ("SELECT 'it''s; ok' AS text;", "SELECT 'it''s; ok' AS text"),
            ("SELECT 'a''''b' AS text;", "SELECT 'a''''b' AS text"),
            (
                "SELECT \"a\"\"b\" FROM \"t\";",
                "SELECT \"a\"\"b\" FROM \"t\"",
            ),
            (
                "SELECT \"a\"\"\"\"b\" FROM \"t\";",
                "SELECT \"a\"\"\"\"b\" FROM \"t\"",
            ),
            (
                "SELECT \"semi;colon\" FROM \"remote.table\";",
                "SELECT \"semi;colon\" FROM \"remote.table\"",
            ),
        ] {
            let source = FlightSqlSource::query(input).unwrap();
            assert_eq!(source, FlightSqlSource::Query(expected.into()));
            assert_eq!(source.clone().normalized().unwrap(), source);
            assert_eq!(FlightSqlSource::query(expected).unwrap(), source);
        }
        assert_eq!(
            FlightSqlSource::query("SELECT 1 AS id;").unwrap(),
            FlightSqlSource::query("SELECT 1 AS id").unwrap()
        );
    }

    #[test]
    fn invalid_query_sources_are_rejected() {
        for input in [
            "",
            " \t\n ",
            "-- no query",
            "SELECT 1; SELECT 2",
            "DELETE FROM t",
            "CREATE TABLE t (id BIGINT)",
            "EXPLAIN SELECT 1",
            "SELECT (",
        ] {
            assert!(matches!(
                FlightSqlSource::query(input),
                Err(DataFusionError::Plan(_))
            ));
            assert!(FlightSqlSource::Query(input.into()).normalized().is_err());
        }
    }

    #[test]
    fn query_discovery_adds_only_an_outer_bound() {
        for (input, expected) in [
            (
                "SELECT 1 AS id;",
                "SELECT * FROM (SELECT 1 AS id) LIMIT 1",
            ),
            (
                "SELECT COUNT(*) AS n FROM t",
                "SELECT * FROM (SELECT COUNT(*) AS n FROM t) LIMIT 1",
            ),
            (
                "WITH x AS (SELECT id FROM t LIMIT 5) SELECT id FROM x ORDER BY id LIMIT 2",
                "SELECT * FROM (WITH x AS (SELECT id FROM t LIMIT 5) SELECT id FROM x ORDER BY id LIMIT 2) LIMIT 1",
            ),
        ] {
            assert_eq!(
                discovery_sql(&FlightSqlSource::query(input).unwrap()),
                expected
            );
        }
    }

    #[test]
    fn scan_projection_preserves_order_and_empty_output() {
        let source = FlightSqlSource::table("t").unwrap();
        for (projection, expected) in [
            (None, "SELECT * FROM \"t\""),
            (Some(vec![2, 0]), "SELECT \"c\", \"a\" FROM \"t\""),
            (Some(vec![]), "SELECT 1 FROM \"t\""),
        ] {
            assert_eq!(
                scan_sql(&source, &schema(), projection.as_deref(), &[], None,).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn scan_projection_quotes_canonical_field_names() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("order", DataType::Int64, false),
            Field::new("a.b", DataType::Int64, false),
            Field::new("a\"b", DataType::Int64, false),
            Field::new("a\"\"b", DataType::Int64, false),
            Field::new("caf\u{e9}", DataType::Int64, false),
        ]));
        assert_eq!(
            scan_sql(
                &FlightSqlSource::table("a.b").unwrap(),
                &schema,
                Some(&[4, 3, 2, 1, 0]),
                &[],
                None,
            )
            .unwrap(),
            "SELECT \"caf\u{e9}\", \"a\"\"\"\"b\", \"a\"\"b\", \"a.b\", \"order\" FROM \"a.b\""
        );
    }

    #[test]
    fn invalid_projection_indices_are_errors() {
        for projection in [vec![3], vec![0, usize::MAX]] {
            assert!(matches!(
                scan_sql(
                    &FlightSqlSource::table("t").unwrap(),
                    &schema(),
                    Some(&projection),
                    &[],
                    None,
                ),
                Err(DataFusionError::Plan(_))
            ));
        }
    }

    #[test]
    fn scan_limits_preserve_the_full_usize_range() {
        for (limit, suffix) in [
            (None, String::new()),
            (Some(0), " LIMIT 0".into()),
            (Some(7), " LIMIT 7".into()),
            (Some(usize::MAX), format!(" LIMIT {}", usize::MAX)),
        ] {
            assert_eq!(
                scan_sql(
                    &FlightSqlSource::table("t").unwrap(),
                    &schema(),
                    None,
                    &[],
                    limit,
                )
                .unwrap(),
                format!("SELECT * FROM \"t\"{suffix}")
            );
        }
    }

    #[test]
    fn query_scan_preserves_nested_queries_order_and_limits() {
        for (input, expected) in [
            (
                "WITH x AS (SELECT a, b, c FROM t LIMIT 5) SELECT * FROM x ORDER BY c LIMIT 2;",
                "SELECT \"c\", \"a\" FROM (WITH x AS (SELECT a, b, c FROM t LIMIT 5) SELECT * FROM x ORDER BY c LIMIT 2) LIMIT 3",
            ),
            (
                "SELECT * FROM (SELECT a, b, c FROM t LIMIT 5) AS q ORDER BY c LIMIT 2;",
                "SELECT \"c\", \"a\" FROM (SELECT * FROM (SELECT a, b, c FROM t LIMIT 5) AS q ORDER BY c LIMIT 2) LIMIT 3",
            ),
        ] {
            assert_eq!(
                scan_sql(
                    &FlightSqlSource::query(input).unwrap(),
                    &schema(),
                    Some(&[2, 0]),
                    &[],
                    Some(3),
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn predicate_renderable_expressions_are_inexact() {
        let source = FlightSqlSource::table("t").unwrap();
        let options = FlightSqlTableOptions::default().with_datafusion_compatible_pushdown();
        for filter in [
            col("a").gt(col("c")),
            col("text").eq(lit("x")),
            col("float").gt(lit(1.5)),
            Expr::Cast(Cast::new(Box::new(col("a")), DataType::Float64)).gt(lit(1.5)),
            abs(col("a")).gt(lit(1_i64)),
            col("a").is_null().or(col("a").gt(lit(1_i64))).not(),
        ] {
            assert_eq!(
                classify_filter(&source, &options, &filter).unwrap(),
                TableProviderFilterPushDown::Inexact,
                "{filter}"
            );
        }
    }

    #[test]
    fn predicate_function_policy_rejects_whole_trees() {
        let source = FlightSqlSource::table("t").unwrap();
        let options = FlightSqlTableOptions::default().with_datafusion_compatible_pushdown();
        let supported = abs(col("a")).gt(lit(1_i64));
        let unsupported = sin(col("a")).gt(lit(0.5));
        for filter in [
            unsupported.clone(),
            supported.clone().and(unsupported.clone()),
            supported.clone().or(unsupported.clone()),
            supported.clone().or(unsupported).not(),
        ] {
            assert_eq!(
                classify_filter(&source, &options, &filter).unwrap(),
                TableProviderFilterPushDown::Unsupported,
                "{filter}"
            );
        }
        for (source, options) in [
            (source, FlightSqlTableOptions::default()),
            (FlightSqlSource::query("SELECT * FROM t").unwrap(), options),
        ] {
            assert_eq!(
                classify_filter(&source, &options, &supported).unwrap(),
                TableProviderFilterPushDown::Unsupported
            );
        }
    }

    #[test]
    fn predicate_rendering_preserves_precedence() {
        let filter = col("a").gt(lit(1_i64)).or(col("b").is_null()).not();
        let sql = scan_sql(
            &FlightSqlSource::table("t").unwrap(),
            &schema(),
            Some(&[0]),
            &[filter, col("c").lt_eq(lit(40_i64))],
            None,
        )
        .unwrap();
        assert_eq!(
            sql,
            "SELECT \"a\" FROM \"t\" WHERE (NOT ((\"a\" > 1) OR \"b\" IS NULL) AND (\"c\" <= 40))"
        );
    }

    #[test]
    fn predicate_rendering_keeps_column_names_literal() {
        for (name, quoted) in [
            ("select", "\"select\""),
            ("a.b", "\"a.b\""),
            ("a\"b", "\"a\"\"b\""),
            ("a\"\"b", "\"a\"\"\"\"b\""),
            ("a\\\"b", "\"a\\\"\"b\""),
            ("caf\u{e9}", "\"caf\u{e9}\""),
        ] {
            let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
            let filter = Expr::Column(Column::new_unqualified(name)).eq(lit(1_i64));
            let sql = scan_sql(
                &FlightSqlSource::table("t").unwrap(),
                &schema,
                None,
                &[filter],
                None,
            )
            .unwrap();
            assert_eq!(sql, format!("SELECT * FROM \"t\" WHERE ({quoted} = 1)"));
        }
    }
}
