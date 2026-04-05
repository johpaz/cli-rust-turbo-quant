//! HTTP Server for TurboQuant API
//!
//! Provides REST API endpoints for text generation with TurboQuant KV cache compression.
//! Compatible with OpenAI API format for easy integration.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

use crate::generation::{create_gemma_tokenizer, GemmaGGUF, LogitsSampler, GemmaTokenizer};

// ─── Shared Application State ──────────────────────────────────────

pub struct AppState {
    pub model: GemmaGGUF,
    pub tokenizer: GemmaTokenizer,
    pub device: Device,
}

pub type SharedState = Arc<Mutex<AppState>>;

// ─── Request/Response Types ────────────────────────────────────────

/// OpenAI-compatible message format
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

/// Simple completion request
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

/// Choice in chat completion response
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Usage information
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Choice in completion response
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

/// Simple completion response
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// Model info response
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}

// ─── Default Values ────────────────────────────────────────────────

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

// ─── Helper Functions ──────────────────────────────────────────────

/// Convert chat messages to a single prompt string
fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<start_of_turn>user\n{}\n<end_of_turn>\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<start_of_turn>user\n{}\n<end_of_turn>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<start_of_turn>model\n{}\n<end_of_turn>\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("{}\n", msg.content));
            }
        }
    }
    // Add final assistant prompt
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Generate text using the model
fn generate_text(
    state: &mut AppState,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<(String, usize, usize), String> {
    // Encode prompt
    let prompt_tokens = state.tokenizer.encode(prompt, true)
        .map_err(|e| format!("Tokenizer encode error: {}", e))?;
    let prompt_len = prompt_tokens.len();
    info!("Prompt tokenized: {} tokens", prompt_len);

    // Get EOS token ID from tokenizer
    let eos_token_id = state.tokenizer.eos_token_id;

    // Create sampler
    let temp = if temperature > 0.0 {
        Some(temperature as f64)
    } else {
        None
    };
    let top_p_val = if top_p > 0.0 && top_p < 1.0 {
        Some(top_p as f64)
    } else {
        None
    };
    let mut sampler = LogitsSampler::new(42, temp, top_p_val);

    // Reset model caches
    state.model.reset_caches();

    // Generate
    let generated_tokens = state.model.generate_text(
        &prompt_tokens,
        max_tokens,
        &mut sampler,
        Some(eos_token_id),
    ).map_err(|e| format!("Generation error: {}", e))?;

    let gen_len = generated_tokens.len();

    // Decode
    let generated_text = state.tokenizer.decode(&generated_tokens, true)
        .map_err(|e| format!("Tokenizer decode error: {}", e))?;

    Ok((generated_text, prompt_len, gen_len))
}

fn generate_id() -> String {
    format!("chatcmpl-{}", uuid_simple())
}

fn uuid_simple() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    (0..32)
        .map(|_| rng.random_range(b'a'..=b'z') as char)
        .collect()
}

// ─── API Handlers ──────────────────────────────────────────────────

/// Health check endpoint
async fn health_handler(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.lock().await.model.config.architecture.clone(),
    })
}

/// List models endpoint
async fn models_handler(State(_state): State<SharedState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "turboquant-model".to_string(),
            object: "model".to_string(),
            created: 1700000000,
            owned_by: "turboquant".to_string(),
        }],
    })
}

/// Chat completions endpoint (OpenAI-compatible)
async fn chat_completions_handler(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!(
        "Chat completion request: {} messages, max_tokens={}, temp={}",
        req.messages.len(),
        req.max_tokens,
        req.temperature
    );

    if req.stream {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": {
                    "message": "Streaming is not yet supported",
                    "type": "not_implemented",
                    "param": "stream",
                    "code": null
                }
            })),
        ));
    }

    let prompt = messages_to_prompt(&req.messages);

    let mut state_guard = state.lock().await;

    match generate_text(
        &mut state_guard,
        &prompt,
        req.max_tokens,
        req.temperature,
        req.top_p,
    ) {
        Ok((text, prompt_tokens, completion_tokens)) => {
            info!("Generated {} tokens", completion_tokens);
            Ok(Json(ChatCompletionResponse {
                id: generate_id(),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                model: "turboquant-model".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            }))
        }
        Err(e) => {
            error!("Generation error: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Generation failed: {}", e),
                        "type": "server_error",
                        "param": null,
                        "code": null
                    }
                })),
            ))
        }
    }
}

/// Completions endpoint (simple)
async fn completions_handler(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    info!(
        "Completion request: prompt={}, max_tokens={}",
        req.prompt, req.max_tokens
    );

    let mut state_guard = state.lock().await;

    match generate_text(
        &mut state_guard,
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_p,
    ) {
        Ok((text, prompt_tokens, completion_tokens)) => {
            info!("Generated {} tokens", completion_tokens);
            Ok(Json(CompletionResponse {
                id: generate_id(),
                object: "text_completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                model: "turboquant-model".to_string(),
                choices: vec![CompletionChoice {
                    index: 0,
                    text,
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            }))
        }
        Err(e) => {
            error!("Generation error: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Generation failed: {}", e),
                        "type": "server_error",
                        "param": null,
                        "code": null
                    }
                })),
            ))
        }
    }
}

// ─── Server Startup ────────────────────────────────────────────────

/// Start the HTTP server
pub async fn start_server(
    model_path: &str,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    info!("🚀 Loading model from: {}", model_path);
    let device = Device::Cpu;

    let model = GemmaGGUF::from_gguf(model_path, device.clone())?;
    info!("✅ Model loaded successfully");

    info!("🔤 Loading tokenizer...");
    let tokenizer = create_gemma_tokenizer(model_path).map_err(|e| {
        anyhow::anyhow!(
            "A tokenizer.json file is required for the API server.\n\
             Download it from HuggingFace and place it next to the GGUF file.\n\
             Error: {}",
            e
        )
    })?;
    info!("✅ Tokenizer loaded successfully");

    let state = Arc::new(Mutex::new(AppState {
        model,
        tokenizer,
        device,
    }));

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    info!("🌐 Starting TurboQuant API server on {}", addr);
    info!("📡 Available endpoints:");
    info!("   GET  http://{}/health", addr);
    info!("   GET  http://{}/v1/models", addr);
    info!("   POST http://{}/v1/chat/completions", addr);
    info!("   POST http://{}/v1/completions", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
