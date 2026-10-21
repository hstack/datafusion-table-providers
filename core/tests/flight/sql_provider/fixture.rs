use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, SqlInfo, TicketStatementQuery};
use arrow_flight::{FlightDescriptor, FlightEndpoint, FlightInfo, Ticket};
use async_trait::async_trait;
use datafusion::arrow::array::{BooleanArray, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use futures::{stream, TryStreamExt};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub operation: &'static str,
    pub sql: Option<String>,
    pub ticket_ids: Vec<u64>,
    pub headers: HashMap<String, String>,
}

pub enum MetadataResponse {
    Schema(SchemaRef),
    MissingSchema,
    MalformedSchema,
}

struct TicketData {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

struct FixtureState {
    context: SessionContext,
    requests: Mutex<Vec<RecordedRequest>>,
    tickets: Mutex<HashMap<u64, TicketData>>,
    next_ticket: AtomicU64,
    next_metadata: Mutex<Option<MetadataResponse>>,
}

pub struct SqlProviderFixture {
    endpoint: String,
    state: Arc<FixtureState>,
    shutdown: Option<oneshot::Sender<()>>,
    server: Option<JoinHandle<std::result::Result<(), tonic::transport::Error>>>,
}

impl SqlProviderFixture {
    pub async fn start() -> Result<Self> {
        let context = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Boolean, true),
            Field::new("c", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(BooleanArray::from(vec![
                    Some(true),
                    Some(false),
                    None,
                    Some(true),
                    Some(false),
                ])),
                Arc::new(Int64Array::from(vec![
                    Some(10),
                    None,
                    Some(30),
                    Some(40),
                    Some(50),
                ])),
            ],
        )?;
        context.register_table(
            "t",
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch]])?),
        )?;
        context.register_table("empty", Arc::new(MemTable::try_new(schema, vec![vec![]])?))?;
        let state = Arc::new(FixtureState {
            context,
            requests: Mutex::new(vec![]),
            tickets: Mutex::new(HashMap::new()),
            next_ticket: AtomicU64::new(1),
            next_metadata: Mutex::new(None),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let endpoint = format!("http://{}", listener.local_addr()?);
        let (shutdown, receiver) = oneshot::channel();
        let service = SqlService(state.clone());
        #[allow(clippy::disallowed_methods)] // The fixture owns and joins the server task.
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(FlightServiceServer::new(service))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = receiver.await;
                })
                .await
        });
        let fixture = Self {
            endpoint,
            state,
            shutdown: Some(shutdown),
            server: Some(server),
        };
        tokio::time::timeout(
            Duration::from_secs(5),
            Channel::from_shared(fixture.endpoint.clone())
                .map_err(|error| DataFusionError::External(Box::new(error)))?
                .connect(),
        )
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))?
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        Ok(fixture)
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.state.requests.lock().unwrap().clone()
    }

    pub fn set_next_metadata(&self, response: MetadataResponse) {
        *self.state.next_metadata.lock().unwrap() = Some(response);
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            shutdown.send(()).map_err(|_| {
                DataFusionError::Execution("SQL fixture stopped before shutdown".into())
            })?;
        }
        if let Some(mut server) = self.server.take() {
            let result = tokio::time::timeout(Duration::from_secs(5), &mut server).await;
            if result.is_err() {
                server.abort();
                let _ = server.await;
            }
            result
                .map_err(|error| DataFusionError::External(Box::new(error)))?
                .map_err(|error| DataFusionError::External(Box::new(error)))?
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
        }
        Ok(())
    }
}

impl Drop for SqlProviderFixture {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(server) = self.server.take() {
            server.abort();
        }
    }
}

struct SqlService(Arc<FixtureState>);

fn headers<T>(request: &Request<T>) -> HashMap<String, String> {
    ["authorization", "x-fixture-scope"]
        .into_iter()
        .filter_map(|name| {
            request
                .metadata()
                .get(name)
                .map(|value| (name.to_owned(), value.to_str().unwrap().to_owned()))
        })
        .collect()
}

#[async_trait]
impl FlightSqlService for SqlService {
    type FlightService = Self;

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<FlightInfo>, Status> {
        let id = self.0.next_ticket.fetch_add(1, Ordering::Relaxed);
        self.0.requests.lock().unwrap().push(RecordedRequest {
            operation: "GetFlightInfo",
            sql: Some(query.query.clone()),
            ticket_ids: vec![id],
            headers: headers(&request),
        });
        let frame = self
            .0
            .context
            .sql(&query.query)
            .await
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let schema = Arc::new(frame.schema().as_arrow().clone());
        let batches = frame
            .collect()
            .await
            .map_err(|error| Status::internal(error.to_string()))?;
        let total_records = batches.iter().map(RecordBatch::num_rows).sum::<usize>() as i64;
        let mut info = FlightInfo::new()
            .try_with_schema(schema.as_ref())
            .map_err(|error| Status::internal(error.to_string()))?
            .with_total_records(total_records);
        self.0
            .tickets
            .lock()
            .unwrap()
            .insert(id, TicketData { schema, batches });
        match self.0.next_metadata.lock().unwrap().take() {
            Some(MetadataResponse::Schema(schema)) => {
                info = info
                    .try_with_schema(schema.as_ref())
                    .map_err(|error| Status::internal(error.to_string()))?;
            }
            Some(MetadataResponse::MissingSchema) => info.schema = Default::default(),
            Some(MetadataResponse::MalformedSchema) => info.schema = vec![0xff].into(),
            None => {}
        }
        let ticket = TicketStatementQuery {
            statement_handle: id.to_le_bytes().to_vec().into(),
        };
        Ok(Response::new(info.with_endpoint(
            FlightEndpoint::new().with_ticket(Ticket::new(ticket.as_any().encode_to_vec())),
        )))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        request: Request<Ticket>,
    ) -> std::result::Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let bytes = ticket
            .statement_handle
            .as_ref()
            .try_into()
            .map_err(|_| Status::invalid_argument("invalid fixture ticket"))?;
        let id = u64::from_le_bytes(bytes);
        self.0.requests.lock().unwrap().push(RecordedRequest {
            operation: "DoGet",
            sql: None,
            ticket_ids: vec![id],
            headers: headers(&request),
        });
        let data = self
            .0
            .tickets
            .lock()
            .unwrap()
            .remove(&id)
            .ok_or_else(|| Status::not_found("unknown or consumed fixture ticket"))?;
        let stream = FlightDataEncoderBuilder::default()
            .with_schema(data.schema)
            .build(stream::iter(data.batches.into_iter().map(Ok)))
            .map_err(|error| Status::internal(error.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}
