use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, SqlInfo, TicketStatementQuery};
use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, Location,
    Ticket,
};
use async_trait::async_trait;
use datafusion::arrow::array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::{oneshot, Notify};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status, Streaming};

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub operation: &'static str,
    pub sql: Option<String>,
    pub ticket_ids: Vec<u64>,
    pub headers: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub enum MetadataResponse {
    Schema(SchemaRef),
    MissingSchema,
    MalformedSchema,
}

#[derive(Debug, Default)]
pub struct RequestGate {
    entered: Notify,
    released: Notify,
}

impl RequestGate {
    pub async fn wait_entered(&self) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(5), self.entered.notified())
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))
    }

    pub fn release(&self) {
        self.released.notify_one();
    }

    async fn block(&self) {
        self.entered.notify_one();
        self.released.notified().await;
    }
}

#[derive(Clone, Debug, Default)]
pub struct Submission {
    pub metadata: Option<MetadataResponse>,
    pub error: Option<Status>,
    pub delay: Duration,
    pub gate: Option<Arc<RequestGate>>,
    pub locations: Vec<Vec<String>>,
    pub batch_size: Option<usize>,
    pub empty_batches: bool,
    pub expired: bool,
    pub stream_failure: Option<Arc<RequestGate>>,
    pub do_get_delay: Duration,
    pub metadata_padding: usize,
    pub ticket_padding: usize,
}

struct TicketData {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    authorization: Option<String>,
    expired: bool,
    stream_failure: Option<Arc<RequestGate>>,
    delay: Duration,
}

struct FixtureState {
    context: SessionContext,
    requests: Mutex<Vec<RecordedRequest>>,
    tickets: Mutex<HashMap<u64, TicketData>>,
    next_ticket: AtomicU64,
    next_session: AtomicU64,
    handshake: Mutex<(Duration, Option<Status>)>,
    handshake_payload_size: AtomicUsize,
    next_submission: Mutex<Submission>,
    submissions: Mutex<HashMap<String, Submission>>,
    completed_sql: Mutex<Vec<String>>,
    require_ticket_auth: AtomicBool,
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
        let string_schema = Arc::new(Schema::new(vec![Field::new(
            "a\"\"b",
            DataType::Utf8,
            false,
        )]));
        context.register_batch(
            "literal_names",
            RecordBatch::try_new(
                string_schema,
                vec![Arc::new(StringArray::from(vec!["a''b", "a'b", "x"]))],
            )?,
        )?;
        Self::start_with_context(context).await
    }

    pub async fn start_with_context(context: SessionContext) -> Result<Self> {
        let state = Arc::new(FixtureState {
            context,
            requests: Mutex::new(vec![]),
            tickets: Mutex::new(HashMap::new()),
            next_ticket: AtomicU64::new(1),
            next_session: AtomicU64::new(1),
            handshake: Mutex::new((Duration::ZERO, None)),
            handshake_payload_size: AtomicUsize::new(0),
            next_submission: Mutex::new(Submission::default()),
            submissions: Mutex::new(HashMap::new()),
            completed_sql: Mutex::new(vec![]),
            require_ticket_auth: AtomicBool::new(false),
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
        self.state.next_submission.lock().unwrap().metadata = Some(response);
    }

    pub fn set_next_submission(&self, submission: Submission) {
        *self.state.next_submission.lock().unwrap() = submission;
    }

    pub fn set_submission(&self, sql: &str, submission: Submission) {
        self.state
            .submissions
            .lock()
            .unwrap()
            .insert(sql.to_owned(), submission);
    }

    pub fn set_handshake(&self, delay: Duration, error: Option<Status>) {
        *self.state.handshake.lock().unwrap() = (delay, error);
    }

    pub fn set_handshake_payload_size(&self, size: usize) {
        self.state
            .handshake_payload_size
            .store(size, Ordering::Relaxed);
    }

    pub fn require_ticket_auth(&self) {
        self.state
            .require_ticket_auth
            .store(true, Ordering::Relaxed);
    }

    pub fn completed_sql(&self) -> Vec<String> {
        self.state.completed_sql.lock().unwrap().clone()
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
    ["authorization", "x-fixture-scope", "x-fixture-default"]
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

    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> std::result::Result<
        Response<
            Pin<Box<dyn Stream<Item = std::result::Result<HandshakeResponse, Status>> + Send>>,
        >,
        Status,
    > {
        self.0.requests.lock().unwrap().push(RecordedRequest {
            operation: "Handshake",
            sql: None,
            ticket_ids: vec![],
            headers: headers(&request),
        });
        let (delay, error) = self.0.handshake.lock().unwrap().clone();
        tokio::time::sleep(delay).await;
        if let Some(error) = error {
            return Err(error);
        }
        let session = self.0.next_session.fetch_add(1, Ordering::Relaxed);
        let size = self.0.handshake_payload_size.load(Ordering::Relaxed);
        let payload = (size > 0).then(|| {
            Ok(HandshakeResponse {
                protocol_version: 0,
                payload: vec![0; size].into(),
            })
        });
        let mut response = Response::new(Box::pin(stream::iter(payload))
            as Pin<Box<dyn Stream<Item = std::result::Result<HandshakeResponse, Status>> + Send>>);
        response.metadata_mut().insert(
            "authorization",
            format!("Bearer fixture-session-{session}").parse().unwrap(),
        );
        Ok(response)
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<FlightInfo>, Status> {
        let submission = self
            .0
            .submissions
            .lock()
            .unwrap()
            .remove(&query.query)
            .unwrap_or_else(|| std::mem::take(&mut *self.0.next_submission.lock().unwrap()));
        let locations = if submission.locations.is_empty() {
            vec![vec![]]
        } else {
            submission.locations.clone()
        };
        let ids = locations
            .iter()
            .map(|_| self.0.next_ticket.fetch_add(1, Ordering::Relaxed))
            .collect::<Vec<_>>();
        let request_headers = headers(&request);
        self.0.requests.lock().unwrap().push(RecordedRequest {
            operation: "GetFlightInfo",
            sql: Some(query.query.clone()),
            ticket_ids: ids.clone(),
            headers: request_headers.clone(),
        });
        if let Some(gate) = &submission.gate {
            gate.block().await;
        }
        tokio::time::sleep(submission.delay).await;
        if let Some(error) = submission.error {
            return Err(error);
        }
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
            .with_total_records(total_records)
            .with_app_metadata(vec![0; submission.metadata_padding]);
        match submission.metadata {
            Some(MetadataResponse::Schema(schema)) => {
                info = info
                    .try_with_schema(schema.as_ref())
                    .map_err(|error| Status::internal(error.to_string()))?;
            }
            Some(MetadataResponse::MissingSchema) => info.schema = Default::default(),
            Some(MetadataResponse::MalformedSchema) => info.schema = vec![0xff].into(),
            None => {}
        }
        let mut partitions = vec![vec![]; locations.len()];
        let batches = batches.into_iter().flat_map(|batch| {
            if let Some(size) = submission.batch_size {
                (0..batch.num_rows())
                    .step_by(size)
                    .map(|start| batch.slice(start, size.min(batch.num_rows() - start)))
                    .collect::<Vec<_>>()
            } else {
                vec![batch]
            }
        });
        for (index, batch) in batches.enumerate() {
            partitions[index % locations.len()].push(batch);
        }
        for ((id, locations), mut batches) in ids.into_iter().zip(locations).zip(partitions) {
            if submission.empty_batches {
                batches.insert(0, RecordBatch::new_empty(schema.clone()));
            }
            self.0.tickets.lock().unwrap().insert(
                id,
                TicketData {
                    schema: schema.clone(),
                    batches,
                    authorization: self
                        .0
                        .require_ticket_auth
                        .load(Ordering::Relaxed)
                        .then(|| request_headers.get("authorization").cloned())
                        .flatten(),
                    expired: submission.expired,
                    stream_failure: submission.stream_failure.clone(),
                    delay: submission.do_get_delay,
                },
            );
            let mut handle = id.to_le_bytes().to_vec();
            handle.resize(8 + submission.ticket_padding, 0);
            let ticket = TicketStatementQuery {
                statement_handle: handle.into(),
            };
            let mut endpoint =
                FlightEndpoint::new().with_ticket(Ticket::new(ticket.as_any().encode_to_vec()));
            endpoint.location = locations.into_iter().map(|uri| Location { uri }).collect();
            info = info.with_endpoint(endpoint);
        }
        self.0.completed_sql.lock().unwrap().push(query.query);
        Ok(Response::new(info))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        request: Request<Ticket>,
    ) -> std::result::Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let bytes = ticket
            .statement_handle
            .get(..8)
            .ok_or_else(|| Status::invalid_argument("invalid fixture ticket"))?
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
        if data.expired {
            return Err(Status::failed_precondition("fixture ticket expired"));
        }
        if let Some(expected) = &data.authorization {
            if headers(&request).get("authorization") != Some(expected) {
                return Err(Status::unauthenticated("wrong token for fixture ticket"));
            }
        }
        tokio::time::sleep(data.delay).await;
        let stream = FlightDataEncoderBuilder::default()
            .with_schema(data.schema)
            .build(stream::iter(data.batches.into_iter().map(Ok)))
            .map_err(|error| Status::internal(error.to_string()));
        let failure = stream::iter(data.stream_failure).then(|gate| async move {
            gate.block().await;
            Err(Status::data_loss("fixture midstream failure"))
        });
        Ok(Response::new(Box::pin(stream.chain(failure))))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}
