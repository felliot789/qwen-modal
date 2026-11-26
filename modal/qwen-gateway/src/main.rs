use axum::{
    Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, HeaderName, Method, StatusCode, Uri, header},
    response::{IntoResponse, Response},
    routing::get,
};
use reqwest::Client;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{net::TcpListener, time::Instant};

const UPSTREAM: &str = "https://makunku3--qwen38-27b-server.us-east.modal.direct";
const LISTEN: &str = "127.0.0.1:20129";
const MAX_BODY: usize = 16 * 1024 * 1024;

#[derive(Clone)]
struct Gateway {
    upstream: String,
    client: Client,
    state_path: PathBuf,
    retry_budget: Duration,
    retry_delay: Duration,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Q8,
    Iq4,
}

impl Variant {
    fn model(self) -> &'static str {
        match self {
            Self::Q8 => "qwen38-hauhau-q8kp",
            Self::Iq4 => "qwen38-hauhau-iq4xs",
        }
    }
}

fn read_variant(path: &Path) -> Option<Variant> {
    match std::fs::read_to_string(path).ok()?.trim() {
        "q8" => Some(Variant::Q8),
        "iq4" => Some(Variant::Iq4),
        _ => None,
    }
}

fn model_matches(path: &str, body: &[u8], variant: Variant) -> Result<(), (StatusCode, String)> {
    if matches!(
        path.trim_end_matches('/'),
        "/v1/chat/completions" | "/v1/messages" | "/v1/completions" | "/v1/responses"
    ) {
        let request: serde_json::Value = serde_json::from_slice(body)
            .map_err(|_| (StatusCode::BAD_REQUEST, "Malformed JSON request".to_owned()))?;
        if request.get("model").and_then(|m| m.as_str()) != Some(variant.model()) {
            return Err((
                StatusCode::CONFLICT,
                format!("Active Qwen variant only accepts model {}", variant.model()),
            ));
        }
    }
    Ok(())
}

fn gateway_client() -> Result<Client, reqwest::Error> {
    Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .no_proxy()
        .build()
}

fn router(gateway: Gateway) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .fallback(forward)
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(gateway)
}

fn strip_hop_headers(headers: &mut HeaderMap) {
    let nominated: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in nominated {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-connection",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

fn authenticated(headers: &HeaderMap) -> bool {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .is_some_and(|(scheme, token)| {
            scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()
        });
    let token_pair = ["modal-key", "modal-secret"].iter().all(|name| {
        headers
            .get(*name)
            .is_some_and(|value| !value.as_bytes().is_empty())
    });
    bearer || token_pair
}

async fn forward(
    State(state): State<Gateway>,
    method: Method,
    uri: Uri,
    mut headers: HeaderMap,
    body: Bytes,
) -> Response {
    strip_hop_headers(&mut headers);
    headers.remove(header::HOST);
    headers.remove(header::CONTENT_LENGTH);
    if !authenticated(&headers) {
        return (StatusCode::UNAUTHORIZED, "Modal credentials required").into_response();
    }

    let Some(variant) = read_variant(&state.state_path) else {
        return (
            StatusCode::CONFLICT,
            "Qwen deployment stopped; run qwen-modal start",
        )
            .into_response();
    };
    let path = uri.path_and_query().map_or("/", |path| path.as_str());
    let Ok(url) = reqwest::Url::parse(&format!("{}{}", state.upstream, path)) else {
        return (StatusCode::BAD_REQUEST, "Invalid request path").into_response();
    };
    if let Err(error) = model_matches(url.path(), &body, variant) {
        return error.into_response();
    }
    let deadline = Instant::now() + state.retry_budget;
    let mut delay = state.retry_delay;

    loop {
        if read_variant(&state.state_path) != Some(variant) {
            return (
                StatusCode::CONFLICT,
                "Qwen deployment stopped or variant changed during retry",
            )
                .into_response();
        }
        let response = match state
            .client
            .request(method.clone(), url.clone())
            .headers(headers.clone())
            .body(body.clone())
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => return (StatusCode::BAD_GATEWAY, "Modal request failed").into_response(),
        };

        // Modal returns 503 while a zero-replica server wakes. Only retry before
        // exposing any response to the caller; a started stream is never replayed.
        if response.status() == StatusCode::SERVICE_UNAVAILABLE && Instant::now() + delay < deadline
        {
            drop(response);
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(2));
            continue;
        }

        let status = response.status();
        let mut headers = response.headers().clone();
        strip_hop_headers(&mut headers);
        let mut downstream = Response::new(Body::from_stream(response.bytes_stream()));
        *downstream.status_mut() = status;
        *downstream.headers_mut() = headers;
        return downstream;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
    let gateway = Gateway {
        upstream: UPSTREAM.into(),
        client: gateway_client()?,
        state_path: PathBuf::from(home).join(".local/state/qwen-modal/variant"),
        retry_budget: Duration::from_secs(180),
        retry_delay: Duration::from_secs(1),
    };
    let listener = TcpListener::bind(LISTEN).await?;
    eprintln!("qwen-modal-gateway listening at http://{LISTEN}");
    axum::serve(listener, router(gateway)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{StreamExt, stream};
    use std::{
        convert::Infallible,
        fs::OpenOptions,
        io::Write,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    struct TestState(PathBuf);

    impl TestState {
        fn new(variant: &str) -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "qwen-gateway-test-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::SeqCst)
            ));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.write_all(variant.as_bytes()).unwrap();
            Self(path)
        }
    }

    impl Drop for TestState {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    async fn serve(app: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn cold_start_retry_and_streaming() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (release, proceed) = tokio::sync::oneshot::channel();
        let proceed = Arc::new(Mutex::new(Some(proceed)));
        let mock = Router::new().fallback({
            let calls = calls.clone();
            move |method: Method, uri: Uri, headers: HeaderMap, body: Bytes| {
                let calls = calls.clone();
                let proceed = proceed.clone();
                async move {
                    assert_eq!(method, Method::POST);
                    assert_eq!(uri.to_string(), "/v1/chat/completions?test=1");
                    assert_eq!(headers[header::AUTHORIZATION], "Bearer test-token");
                    assert!(!headers.contains_key("x-hop-only"));
                    assert_eq!(body, "{\"model\":\"qwen38-hauhau-q8kp\",\"stream\":true}");
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        return (StatusCode::SERVICE_UNAVAILABLE, "cold").into_response();
                    }
                    let proceed = proceed.lock().unwrap().take().unwrap();
                    let chunks =
                        stream::iter([Ok::<_, Infallible>(Bytes::from_static(b"data: first\n\n"))])
                            .chain(stream::once(async move {
                                proceed.await.unwrap();
                                Ok(Bytes::from_static(b"data: [DONE]\n\n"))
                            }));
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(chunks))
                        .unwrap()
                }
            }
        });
        let upstream = serve(mock).await;
        let client = gateway_client().unwrap();
        let state_path = TestState::new("q8\n");
        let proxy = Gateway {
            upstream,
            client: client.clone(),
            state_path: state_path.0.clone(),
            retry_budget: Duration::from_secs(2),
            retry_delay: Duration::from_millis(10),
        };
        let local = serve(router(proxy)).await;
        assert_eq!(
            client
                .get(format!("{local}/healthz"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            client
                .post(format!("{local}/v1/chat/completions"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let response = client
            .post(format!("{local}/v1/chat/completions?test=1"))
            .header(header::AUTHORIZATION, "Bearer test-token")
            .header(header::CONNECTION, "x-hop-only")
            .header("x-hop-only", "ignored")
            .body("{\"model\":\"qwen38-hauhau-q8kp\",\"stream\":true}")
            .send()
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let mut body = response.bytes_stream();
        let first = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first, "data: first\n\n");
        release.send(()).unwrap();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: [DONE]\n\n");
        assert!(body.next().await.is_none());
    }

    #[tokio::test]
    async fn failed_stream_is_not_replayed() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (release, proceed) = tokio::sync::oneshot::channel();
        let proceed = Arc::new(Mutex::new(Some(proceed)));
        let upstream = serve(Router::new().fallback({
            let calls = calls.clone();
            move || {
                let calls = calls.clone();
                let proceed = proceed.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    let proceed = proceed.lock().unwrap().take().unwrap();
                    let chunks = stream::iter([Ok(Bytes::from_static(b"data: first\n\n"))]).chain(
                        stream::once(async move {
                            proceed.await.unwrap();
                            Err(std::io::Error::other("upstream disconnected"))
                        }),
                    );
                    Response::new(Body::from_stream(chunks))
                }
            }
        }))
        .await;
        let client = gateway_client().unwrap();
        let state_path = TestState::new("q8\n");
        let local = serve(router(Gateway {
            upstream,
            client: client.clone(),
            state_path: state_path.0.clone(),
            retry_budget: Duration::from_secs(2),
            retry_delay: Duration::from_millis(10),
        }))
        .await;
        let mut body = client
            .get(local)
            .header(header::AUTHORIZATION, "Bearer test-token")
            .send()
            .await
            .unwrap()
            .bytes_stream();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: first\n\n");
        release.send(()).unwrap();
        assert!(body.next().await.unwrap().is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_stops_when_caller_disconnects() {
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(tokio::sync::Notify::new());
        let upstream = serve(Router::new().fallback({
            let calls = calls.clone();
            let started = started.clone();
            move || {
                let calls = calls.clone();
                let started = started.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    started.notify_one();
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }
        }))
        .await;
        let client = gateway_client().unwrap();
        let state_path = TestState::new("q8\n");
        let local = serve(router(Gateway {
            upstream,
            client: client.clone(),
            state_path: state_path.0.clone(),
            retry_budget: Duration::from_secs(5),
            retry_delay: Duration::from_millis(100),
        }))
        .await;
        let task = tokio::spawn(async move {
            client
                .get(local)
                .header(header::AUTHORIZATION, "Bearer test-token")
                .send()
                .await
        });
        started.notified().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::sleep(Duration::from_millis(350)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stopped_and_mismatched_profiles_never_reach_upstream() {
        let calls = Arc::new(AtomicUsize::new(0));
        let upstream = serve(Router::new().fallback({
            let calls = calls.clone();
            move || {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    StatusCode::OK
                }
            }
        }))
        .await;
        let client = gateway_client().unwrap();
        let state_path = TestState::new("q8\n");
        let local = serve(router(Gateway {
            upstream,
            client: client.clone(),
            state_path: state_path.0.clone(),
            retry_budget: Duration::from_secs(2),
            retry_delay: Duration::from_millis(10),
        }))
        .await;
        let request = |path: &str, model| {
            client
                .post(format!("{local}{path}"))
                .header(header::AUTHORIZATION, "Bearer test-token")
                .body(format!("{{\"model\":\"{model}\"}}"))
        };

        for path in [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/completions",
            "/v1/responses",
        ] {
            assert_eq!(
                request(path, "qwen38-hauhau-iq4xs")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::CONFLICT
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            request("/v1/chat/completions", "qwen38-hauhau-q8kp")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        std::fs::write(&state_path.0, "iq4\n").unwrap();
        assert_eq!(
            request("/v1/messages", "qwen38-hauhau-q8kp")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            request("/v1/messages", "qwen38-hauhau-iq4xs")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        std::fs::write(&state_path.0, "stopped\n").unwrap();
        assert_eq!(
            request("/v1/models", "qwen38-hauhau-iq4xs")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        std::fs::remove_file(&state_path.0).unwrap();
        assert_eq!(
            request("/v1/models", "qwen38-hauhau-iq4xs")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            client
                .get(format!("{local}/healthz"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_stops_when_profile_is_disabled() {
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(tokio::sync::Notify::new());
        let upstream = serve(Router::new().fallback({
            let calls = calls.clone();
            let started = started.clone();
            move || {
                let calls = calls.clone();
                let started = started.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    started.notify_one();
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }
        }))
        .await;
        let client = gateway_client().unwrap();
        let state_path = TestState::new("q8\n");
        let local = serve(router(Gateway {
            upstream,
            client: client.clone(),
            state_path: state_path.0.clone(),
            retry_budget: Duration::from_secs(5),
            retry_delay: Duration::from_millis(100),
        }))
        .await;
        let task = tokio::spawn(async move {
            client
                .get(local)
                .header(header::AUTHORIZATION, "Bearer test-token")
                .send()
                .await
        });
        started.notified().await;
        std::fs::write(&state_path.0, "stopped\n").unwrap();
        assert_eq!(task.await.unwrap().unwrap().status(), StatusCode::CONFLICT);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
