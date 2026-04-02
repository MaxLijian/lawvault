use arrow_array::{Float32Array, StringArray};
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{path::BaseDirectory, AppHandle, Emitter, Manager};

// ==========================================
// 1. 提示词 (Prompts)
// ==========================================

const PLANNER_PROMPT: &str = r#"
你是法律检索规划专家。将用户问题拆解为向量检索任务清单。

数据库说明：
- 包含法律、行政法规、司法解释、地方法规的条文 Embedding
- 检索方式为语义相似度匹配

拆解原则：

1. 提取核心法律概念
将问题中的关键法律术语提取为检索词。
正确："故意伤害罪的量刑标准"
错误："打人会被判多久"（口语化，相似度低）

2. 分层检索
先查主要法律依据，再查司法解释细则。
示例：["劳动合同解除的法定情形", "解除劳动合同的经济补偿标准"]

3. 避免过度拆解
简单问题1个任务即可，复杂问题不超过5个。
问题："诉讼时效是多久" → ["民事诉讼时效期间"]
问题："房屋买卖合同纠纷如何处理" → ["房屋买卖合同违约责任", "房屋买卖合同解除条件", "房屋买卖纠纷管辖规定"]

4. 使用标准法律术语
用"不当得利"而非"多收的钱要还吗"
用"劳动争议仲裁时效"而非"劳动纠纷多久失效"

输出格式：
仅输出 JSON 数组，不含任何其他内容：
["任务1", "任务2", "任务3"]

用户问题："{user_query}"
"#;

const EXECUTOR_PROMPT: &str = r#"
你是检索结果评估器。你的核心职责是：**根据当前的搜索结果，动态修正后续的检索计划**。

上下文：
- 用户原始问题："{user_query}"
- 刚刚执行的任务："{current_task}"

检索结果：
{search_results}

待办任务清单：
{remaining_todo_list}

**决策逻辑（必须严格遵守）**：

1. **评估结果质量**：
   - 如果检索结果为空或完全不相关 -> **必须**在待办清单头部插入一个新的、换了关键词的检索任务（例如将“量刑”改为“刑法 第X条”）。
   - 如果检索结果非常完美 -> 继续执行原定计划。

2. **发现新线索**：
   - 如果检索到的法条提到了一个新的关键法律概念（例如搜“离婚”时发现了“离婚冷静期”规定），且这对回答用户问题很重要 -> **必须**将查询该新概念加入待办清单。

3. **去重与精简**：
   - 检查待办清单，如果后续任务已经被当前的检索结果覆盖了，请将其删除。

**输出格式（仅 JSON）**：
{
  "thought": "深刻分析：刚才搜到了什么？缺什么？为什么要修改（或保持）清单？",
  "new_todo_list": ["任务A", "任务B"...]
}
"#;

// ==========================================
// 2. 数据结构
// ==========================================
#[derive(Serialize, Deserialize, Debug)]
pub struct UserFolder {
    id: i32,
    name: String,
    created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppSettings {
    pub search_top_k: usize,
    pub display_density: String,
    pub embedding_base_url: String,
    pub embedding_api_key: String,
    pub embedding_model: String,
    pub reranker_type: String,
    pub reranker_base_url: String,
    pub reranker_api_key: String,
    pub reranker_model: String,
    pub custom_data_path: Option<String>,
    pub enable_ai_chat: bool,
    pub chat_base_url: String,
    pub chat_api_key: String,
    pub chat_model: String,
    pub chat_top_k: usize,
    #[serde(default = "default_max_loops")]
    pub max_agent_loops: i32,
    #[serde(default)]
    pub use_external_chat_api: bool,
    pub external_chat_base_url: String,
    pub external_chat_api_key: String,
    pub external_chat_model: String,
    pub external_chat_api_choice: i32,
    pub external_chat_base_url_2: String,
    pub external_chat_api_key_2: String,
    pub external_chat_model_2: String,
}

fn default_max_loops() -> i32 {
    5
}

fn default_reranker_model() -> String {
    "ms-marco-MiniLM-L-12-v2".to_string()
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            search_top_k: 50,
            display_density: "comfortable".to_string(),
            custom_data_path: None,
            embedding_base_url: "http://localhost:11434/v1".to_string(),
            embedding_api_key: "ollama".to_string(),
            embedding_model: "qwen3-embedding:0.6b".to_string(),
            reranker_type: "dashscope".to_string(),
            reranker_base_url: "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank".to_string(),
            reranker_api_key: "sk-b5ed0ddae2bb4ca1ae401a29111e0360".to_string(),
            reranker_model: "qwen3-vl-rerank".to_string(),
            enable_ai_chat: false,
            chat_base_url: "http://localhost:11434/v1".to_string(),
            chat_api_key: "ollama".to_string(),
            chat_model: "qwen3".to_string(),
            chat_top_k: 5,
            max_agent_loops: 5,
            use_external_chat_api: false,
            external_chat_base_url: "https://api.minimax.chat/v1".to_string(),
            external_chat_api_key: "".to_string(),
            external_chat_model: "MiniMax-M2.5".to_string(),
            external_chat_api_choice: 1,
            external_chat_base_url_2: "https://longcat.chat/v1".to_string(),
            external_chat_api_key_2: "".to_string(),
            external_chat_model_2: "".to_string(),
        }
    }
}

#[derive(Serialize, Debug)]
struct LawNameSuggestion {
    name: String,
    region: String,
    category: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LawChunk {
    id: String,
    pub _distance: f32,
    pub content: String,
    pub law_name: String,
    category: String,
    publish_date: String,
    part: String,
    chapter: String,
    pub article_number: String,
    region: String,
    source_file: String,
    #[serde(default)]
    pub full_article: Option<String>,
}

// 用户收藏结构体
#[derive(Serialize, Deserialize, Debug)]
pub struct UserFavorite {
    id: i32,
    law_id: String,
    law_name: String,
    article_number: String,
    content: String,
    created_at: String,
    tags: Option<String>,
    folder_id: Option<i32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchHistoryItem {
    id: i32,
    query: String,
    timestamp: i64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CustomTemplate {
    id: i32,
    name: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DraftMaterial {
    id: i32,
    law_id: String,
    law_name: String,
    article_number: String,
    content: String,
    added_at: String,
}

pub struct AppState {
    pub settings: Mutex<AppSettings>,
    pub settings_path: PathBuf,
    pub app_data_dir: PathBuf,
    // 存储 user_data.db 的路径，方便后续连接
    pub user_db_path: PathBuf,
    pub chat_tasks: Mutex<HashMap<String, tauri::async_runtime::JoinHandle<()>>>,
    pub agent_abort_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

// --- Agent 相关结构 ---
#[derive(Serialize, Clone, Debug)]
pub struct AgentUpdateEvent {
    pub step_type: String,
    pub todo_list: Vec<String>,
    pub completed_log: Vec<CompletedTask>,
    pub current_task: Option<String>,
    pub thought: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct CompletedTask {
    pub task: String,
    pub thought: String,
}

#[derive(Deserialize)]
struct ExecutorResponse {
    thought: String,
    new_todo_list: Vec<String>,
}

// ==========================================
// 3. 辅助函数
// ==========================================

// 连接 content.db (法条库)
fn connect_sqlite(data_dir: &std::path::Path) -> Result<Connection, String> {
    let db_path_buf = data_dir.join("content.db");
    let path_str = db_path_buf.to_string_lossy().to_string();

    #[cfg(windows)]
    {
        if path_str.starts_with(r"\\?\") {
            path_str = path_str[4..].to_string();
        }
    }

    Connection::open(path_str).map_err(|e| format!("SQLite connect error: {}", e))
}

// 连接 user_data.db (用户库)
fn connect_user_db(db_path: &PathBuf) -> Result<Connection, String> {
    let conn = Connection::open(db_path).map_err(|e| format!("无法打开用户数据库: {}", e))?;
    conn.execute("CREATE TABLE IF NOT EXISTS favorite_folders (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)", []).map_err(|e| e.to_string())?;
    conn.execute("CREATE TABLE IF NOT EXISTS favorites (id INTEGER PRIMARY KEY AUTOINCREMENT, law_id TEXT UNIQUE, law_name TEXT, article_number TEXT, content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, tags TEXT)", []).map_err(|e| e.to_string())?;

    let column_exists: bool = conn
        .prepare("PRAGMA table_info(favorites)")
        .map_err(|e| e.to_string())?
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name == "folder_id")
        })
        .map_err(|e| e.to_string())?
        .any(|res| res.unwrap_or(false));
    if !column_exists {
        conn.execute("ALTER TABLE favorites ADD COLUMN folder_id INTEGER", [])
            .map_err(|e| e.to_string())?;
    }

    conn.execute("CREATE TABLE IF NOT EXISTS search_history (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT UNIQUE, timestamp INTEGER)", []).map_err(|e| e.to_string())?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS draft_materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            law_id TEXT UNIQUE,
            law_name TEXT,
            article_number TEXT,
            content TEXT,
            added_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS custom_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            content TEXT
        )",
        [],
    )
    .map_err(|e| e.to_string())?;

    Ok(conn)
}
fn load_settings_from_disk(path: &PathBuf) -> AppSettings {
    if let Ok(content) = fs::read_to_string(path) {
        if let Ok(settings) = serde_json::from_str(&content) {
            return settings;
        }
    }
    AppSettings::default()
}

fn get_effective_data_dir(state: &AppState) -> PathBuf {
    let settings = state.settings.lock().unwrap();
    if let Some(custom_path) = &settings.custom_data_path {
        if !custom_path.trim().is_empty() {
            let path = PathBuf::from(custom_path);
            if path.exists() {
                return path;
            }
        }
    }
    state.app_data_dir.clone()
}

async fn get_embedding(
    text: &str,
    base_url: &str,
    api_key: &str,
    model: &str,
) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?;
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let prompt = text.replace("\n", " ");

    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "model": model,
            "input": prompt,
        }))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !res.status().is_success() {
        return Err(format!("Embedding API Error: {}", res.status()));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;

    if let Some(data) = json.get("data") {
        if let Some(first) = data.get(0) {
            if let Some(vec) = first.get("embedding") {
                let embedding: Vec<f32> = vec
                    .as_array()
                    .ok_or("Invalid embedding format")?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                return Ok(embedding);
            }
        }
    }
    if let Some(vec) = json.get("embedding") {
        let embedding: Vec<f32> = vec
            .as_array()
            .ok_or("Invalid embedding format")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        return Ok(embedding);
    }

    Err("Could not find embedding in response".to_string())
}

async fn call_llm(
    model: &str,
    prompt: &str,
    base_url: &str,
    api_key: &str,
    enable_thinking: bool,
) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    fn extract_content(json: &serde_json::Value) -> Option<String> {
        if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
            return Some(content.to_string());
        }
        if let Some(content) = json["choices"][0]["content"].as_str() {
            return Some(content.to_string());
        }
        if let Some(content) = json["message"]["content"].as_str() {
            return Some(content.to_string());
        }
        if let Some(content) = json["content"].as_str() {
            return Some(content.to_string());
        }
        None
    }

    fn clean_text(raw: &str) -> String {
        raw.replace("<longcat_think>", "<think>")
            .replace("</longcat_think>", "</think>")
            .replace("<result>", "")
            .replace("</result>", "")
    }

    let mut req_map = serde_json::Map::new();
    req_map.insert("model".into(), serde_json::Value::String(model.to_string()));
    req_map.insert(
        "messages".into(),
        serde_json::json!([{ "role": "user", "content": prompt }]),
    );
    req_map.insert("temperature".into(), serde_json::json!(0.1));
    req_map.insert("stream".into(), serde_json::json!(false));
    req_map.insert("max_tokens".into(), serde_json::json!(2048));
    if enable_thinking {
        req_map.insert("enable_thinking".into(), serde_json::json!(true));
    }

    let mut last_err = String::new();
    for attempt in 1..=3 {
        let req_body = serde_json::Value::Object(req_map.clone());
        let send_result = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&req_body)
            .send()
            .await;

        match send_result {
            Ok(res) => {
                let status = res.status();
                let body_text = res.text().await.unwrap_or_default();

                if !status.is_success() {
                    last_err = format!("LLM API Error: {} body={}", status, body_text);

                    if status.as_u16() == 400 && req_map.contains_key("max_tokens") {
                        req_map.remove("max_tokens");
                        req_map.insert("max_completion_tokens".into(), serde_json::json!(2048));
                    }

                    if status.is_server_error() || status.as_u16() == 429 {
                        tokio::time::sleep(std::time::Duration::from_secs(2 * attempt)).await;
                        continue;
                    }

                    if attempt < 3 {
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        continue;
                    }
                    return Err(last_err);
                }

                let json: serde_json::Value =
                    serde_json::from_str(&body_text).map_err(|e| format!("JSON parse failed: {} raw={}", e, body_text))?;

                if let Some(content) = extract_content(&json) {
                    return Ok(content);
                }

                last_err = format!("No content in response: {}", json);
            }
            Err(e) => {
                last_err = format!("Request failed: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(2 * attempt)).await;
                continue;
            }
        }
    }

    // 回退到流式模式，兼容部分 LongCat 接口对非流式返回不稳定的情况
    req_map.insert("stream".into(), serde_json::json!(true));
    if req_map.contains_key("max_completion_tokens") {
        req_map.remove("max_completion_tokens");
        req_map.insert("max_tokens".into(), serde_json::json!(2048));
    }

    for attempt in 1..=2 {
        let req_body = serde_json::Value::Object(req_map.clone());
        let send_result = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&req_body)
            .send()
            .await;

        match send_result {
            Ok(res) => {
                let status = res.status();
                if !status.is_success() {
                    let body_text = res.text().await.unwrap_or_default();
                    last_err = format!("LLM Stream API Error: {} body={}", status, body_text);
                    tokio::time::sleep(std::time::Duration::from_secs(attempt as u64)).await;
                    continue;
                }

                let mut stream = res.bytes_stream();
                let mut line_buf = String::new();
                let mut collected = String::new();

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            line_buf.push_str(&text);

                            while let Some(newline_pos) = line_buf.find('\n') {
                                let line = line_buf[..newline_pos].to_string();
                                line_buf = line_buf[newline_pos + 1..].to_string();
                                let trimmed_line = line.trim();

                                if trimmed_line == "[DONE]"
                                    || trimmed_line == "data:[DONE]"
                                    || trimmed_line == "data: [DONE]"
                                {
                                    break;
                                }

                                if trimmed_line.starts_with("data:") && trimmed_line.len() > 5 {
                                    let json_str = trimmed_line[5..].trim_start();
                                    if json_str.is_empty() {
                                        continue;
                                    }

                                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str)
                                    {
                                        if let Some(reasoning) =
                                            json["choices"][0]["delta"]["reasoning_content"].as_str()
                                        {
                                            collected.push_str(&clean_text(reasoning));
                                        }
                                        if let Some(content) =
                                            json["choices"][0]["delta"]["content"].as_str()
                                        {
                                            collected.push_str(&clean_text(content));
                                        } else if let Some(content) = extract_content(&json) {
                                            collected.push_str(&clean_text(&content));
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            last_err = format!("LLM stream read error: {}", e);
                        }
                    }
                }

                if !collected.trim().is_empty() {
                    return Ok(collected);
                }
                last_err = "LLM stream returned empty content".to_string();
            }
            Err(e) => {
                last_err = format!("LLM stream request failed: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(attempt as u64)).await;
                continue;
            }
        }
    }

    Err(last_err)
}

fn parse_executor_response(raw: &str) -> Option<ExecutorResponse> {
    let clean = clean_json_str(raw);

    if let Ok(res) = serde_json::from_str::<ExecutorResponse>(&clean) {
        return Some(res);
    }

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&clean) {
        let thought = v
            .get("thought")
            .and_then(|x| x.as_str())
            .or_else(|| v.get("analysis").and_then(|x| x.as_str()))
            .or_else(|| v.get("reasoning").and_then(|x| x.as_str()))
            .unwrap_or("已完成当前任务评估。")
            .to_string();

        let candidates = ["new_todo_list", "todo_list", "tasks", "next_tasks"];
        for key in candidates {
            if let Some(arr) = v.get(key).and_then(|x| x.as_array()) {
                let list: Vec<String> = arr
                    .iter()
                    .filter_map(|item| item.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect();
                if !list.is_empty() {
                    return Some(ExecutorResponse {
                        thought,
                        new_todo_list: list,
                    });
                }
            }
        }
    }

    let mut thought_line = String::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.starts_with("思考:") || t.starts_with("thought:") || t.starts_with("Thought:") {
            thought_line = t
                .trim_start_matches("思考:")
                .trim_start_matches("thought:")
                .trim_start_matches("Thought:")
                .trim()
                .to_string();
            break;
        }
    }

    if thought_line.is_empty() {
        thought_line = "已完成当前任务评估。".to_string();
    }

    Some(ExecutorResponse {
        thought: thought_line,
        new_todo_list: vec![],
    })
}

fn clean_json_str(s: &str) -> String {
    let mut content = s.to_string();

    // 1. 移除 <think>...</think>
    while let Some(start) = content.find("<think>") {
        if let Some(end) = content.find("</think>") {
            if end > start {
                content.replace_range(start..end + 8, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // 2. 移除 ```think ... ``` 代码块格式
    while let Some(start) = content.find("```think") {
        if let Some(end_block) = content[start + 8..].find("```") {
            let actual_end = start + 8 + end_block + 3;
            content.replace_range(start..actual_end, "");
        } else {
            break;
        }
    }

    // 3. 移除 <result>...</result> 标签
    while let Some(start) = content.find("<result>") {
        if let Some(end) = content.find("</result>") {
            if end > start {
                content.replace_range(start..end + 9, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // 4. 移除 XML 标签 <...>
    let mut i = 0;
    let bytes = content.into_bytes();
    let mut result = Vec::new();
    let mut in_tag = false;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            in_tag = true;
            i += 1;
        } else if bytes[i] == b'>' {
            in_tag = false;
            i += 1;
        } else if !in_tag {
            result.push(bytes[i]);
            i += 1;
        } else {
            i += 1;
        }
    }
    let content = String::from_utf8_lossy(&result).to_string();

    // 5. 移除普通注释
    let content = content.replace("<!--", "").replace("-->", "");

    // 6. 移除 <style>...</style>
    let mut content = content;
    while let Some(start) = content.find("<style>") {
        if let Some(end) = content.find("</style>") {
            if end > start {
                content.replace_range(start..end + 8, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // 7. 移除 <script>...</script>
    while let Some(start) = content.find("<script>") {
        if let Some(end) = content.find("</script>") {
            if end > start {
                content.replace_range(start..end + 9, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // 8. 智能提取 JSON (Array 或 Object)
    let first_brace = content.find('{');
    let first_bracket = content.find('[');

    let (start, end) = match (first_brace, first_bracket) {
        (Some(brace), Some(bracket)) => {
            if brace < bracket {
                (brace, content.rfind('}'))
            } else {
                (bracket, content.rfind(']'))
            }
        },
        (Some(brace), None) => (brace, content.rfind('}')),
        (None, Some(bracket)) => (bracket, content.rfind(']')),
        (None, None) => return content,
    };

    match (start, end) {
        (s, Some(e)) if s <= e => content[s..=e].to_string(),
        _ => content
    }
}

// ==========================================
// 3.5. Rerank — 支持多种 Reranker 服务
// ==========================================

fn rerank_chunks(
    query: &str,
    chunks: Vec<LawChunk>,
    settings: &AppSettings,
) -> Result<Vec<LawChunk>, String> {
    if chunks.len() <= 1 {
        return Ok(chunks);
    }

    let reranker_type = &settings.reranker_type;
    let reranker_base_url = &settings.reranker_base_url;
    let reranker_api_key = &settings.reranker_api_key;
    let reranker_model = &settings.reranker_model;
    let top_n = chunks.len().min(200);

    // 如果是 local 类型，跳过重排序
    if reranker_type == "local" {
        println!("[Rerank] 使用 local 模式，跳过重排序");
        return Ok(chunks);
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?;

    let documents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

    let req_body = match reranker_type.as_str() {
        "dashscope" => {
            // 阿里云 DashScope API 格式
            serde_json::json!({
                "model": reranker_model,
                "input": {
                    "query": query,
                    "documents": documents
                },
                "parameters": {
                    "return_documents": true,
                    "top_n": top_n
                }
            })
        },
        "custom" => {
            // 自定义 API 格式，使用通用的 OpenAI 风格
            serde_json::json!({
                "model": reranker_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": false
            })
        },
        _ => {
            // 默认使用自定义格式
            serde_json::json!({
                "model": reranker_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": false
            })
        }
    };

    let mut request = client.post(reranker_base_url)
        .header("Content-Type", "application/json");

    // 添加 API Key 头部
    if !reranker_api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", reranker_api_key));
    }

    let res = request
        .json(&req_body)
        .send()
        .map_err(|e| format!("Rerank 请求失败: {}", e))?;

    if !res.status().is_success() {
        return Err(format!("Rerank API 错误: {} - {}", res.status(), res.text().unwrap_or("未知错误".to_string())));
    }

    let json: serde_json::Value = res.json().map_err(|e| format!("Rerank 解析失败: {}", e))?;

    let results = match reranker_type.as_str() {
        "dashscope" => {
            // 阿里云 DashScope 响应格式：output.results
            json.get("output")
                .and_then(|o| o.get("results"))
                .and_then(|r| r.as_array())
                .ok_or("Rerank 响应格式错误：缺少 output.results 字段")?
        },
        _ => {
            // 通用格式：results
            json.get("results")
                .and_then(|r| r.as_array())
                .ok_or("Rerank 响应格式错误：缺少 results 字段")?
        }
    };

    let mut reranked: Vec<LawChunk> = Vec::with_capacity(chunks.len());
    let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for item in results {
        let idx = item.get("index")
            .and_then(|v| v.as_u64())
            .and_then(|v| Some(v as usize))
            .unwrap_or(0);
        if idx < chunks.len() && !used.contains(&idx) {
            used.insert(idx);
            reranked.push(chunks[idx].clone());
        }
    }

    // 兜底：没排上的按原顺序追加
    for (i, chunk) in chunks.iter().enumerate() {
        if !used.contains(&i) {
            reranked.push(chunk.clone());
        }
    }

    Ok(reranked)
}

// ==========================================
// 4. 核心逻辑
// ==========================================

// 法律停用词表（用于 FTS5 召回前置过滤）
const LEGAL_STOP_WORDS: &[&str] = &[
    "基于", "关于", "进行", "其中", "及其", "以及", "或者", "并且",
    "可以", "应当", "必须", "不得", "不能", "不要", "相关", "有关",
    "其他", "其它", "上述", "以下", "包括", "除外的", "根据", "按照",
    "通过", "对于", "由于", "因此", "所以", "但是", "然而",
    "其", "这", "那", "这个", "那个", "这些", "那些", "的", "了",
    "在", "是", "有", "和", "与", "或", "不", "也", "都", "而",
];

// 法条编号正则
const ARTICLE_REGEX: &str = r"第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十]?)?(?:第?[一二三四五六七八九十百千零\d]+项?)?";

fn filter_legal_stop_words(query: &str) -> String {
    let words: Vec<&str> = query.split_whitespace().collect();
    let filtered: Vec<&str> = words
        .into_iter()
        .filter(|w| !LEGAL_STOP_WORDS.contains(w))
        .collect();
    filtered.join(" ")
}

fn extract_article_numbers(query: &str) -> Vec<String> {
    let mut articles = Vec::new();
    let regex = regex::Regex::new(ARTICLE_REGEX).ok();
    if let Some(re) = regex {
        for cap in re.find_iter(query) {
            articles.push(cap.as_str().to_string());
        }
    }
    articles
}

fn fts5_search_query(query: &str) -> String {
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.is_empty() {
        return query.to_string();
    }
    let processed: Vec<String> = words
        .iter()
        .filter(|w| w.len() > 1 && !LEGAL_STOP_WORDS.contains(*w))
        .map(|w| format!("\"{}\"", w))
        .collect();
    if processed.len() > 1 {
        processed.join(" OR ")
    } else if processed.len() == 1 {
        processed[0].clone()
    } else {
        query.to_string()
    }
}

// Small-to-Big: 根据 chunk 的 law_name 和 article_number 获取完整法条
fn fetch_full_article(conn: &Connection, law_name: &str, _article_number: &str) -> Option<String> {
    let sql = "SELECT full_text FROM full_texts WHERE law_name = ? LIMIT 1";
    let mut stmt = conn.prepare(sql).ok()?;
    let full_text: String = stmt.query_row([law_name], |row| row.get(0)).ok()?;
    if full_text.is_empty() {
        return None;
    }
    Some(full_text)
}

// 引用校验：检查 LLM 回复中引用的法条是否在 context 中存在
fn validate_citations(llm_response: &str, context_articles: &[&str]) -> Vec<String> {
    let regex = match regex::Regex::new(ARTICLE_REGEX) {
        Ok(re) => re,
        Err(_) => return Vec::new(),
    };
    let mut missing_citations = Vec::new();
    for cap in regex.find_iter(llm_response) {
        let article = cap.as_str();
        let found = context_articles.iter().any(|ctx| ctx.contains(article));
        if !found {
            missing_citations.push(article.to_string());
        }
    }
    missing_citations
}

pub async fn search_law_logic(
    query: String,
    filter_region: Option<String>,
    state: &AppState,
) -> Result<Vec<LawChunk>, String> {
    println!(">>> (Logic) Searching for: {}", query);

    let settings = state.settings.lock().unwrap().clone();
    let data_dir = get_effective_data_dir(state);

    // ========== 三路查询拆解 ==========

    // 路1：正则提取法条编号（精确匹配）
    let article_numbers = extract_article_numbers(&query);
    println!(">>> [三路] 提取到法条编号: {:?}", article_numbers);

    // 路2：向量检索
    let vector = match get_embedding(
        &query,
        &settings.embedding_base_url,
        &settings.embedding_api_key,
        &settings.embedding_model,
    )
    .await {
        Ok(v) => v,
        Err(e) => {
            println!(">>> [降级] 向量检索失败: {}，降级到 FTS5", e);
            Vec::new()
        }
    };

    let lancedb_path_buf = data_dir.join("law_db.lancedb");
    let path_str = lancedb_path_buf.to_string_lossy().to_string();
    #[cfg(windows)]
    {
        if path_str.starts_with(r"\\?\") {
            path_str = path_str[4..].to_string();
        }
    }

    let mut vector_results: Vec<(String, f32)> = Vec::new();

    if !vector.is_empty() && lancedb_path_buf.exists() {
        let db = lancedb::connect(&path_str).execute().await;
        if let Ok(db) = db {
            let table = db.open_table("laws_vectors").execute().await;
            if let Ok(table) = table {
                let fetch_limit = settings.search_top_k * 4;
                let query = table.query().nearest_to(vector);
                if let Ok(query) = query {
                    let results = query.limit(fetch_limit).execute().await;
                    if let Ok(mut stream) = results {
                        while let Some(item) = stream.next().await {
                            if let Ok(batch) = item {
                                let id_col = batch.column_by_name("chunk_id");
                                let dist_col = batch.column_by_name("_distance");
                                if let (Some(id_col), Some(dist_col)) = (id_col, dist_col) {
                                    let ids = id_col.as_any().downcast_ref::<StringArray>();
                                    let dists = dist_col.as_any().downcast_ref::<Float32Array>();
                                    if let (Some(ids), Some(dists)) = (ids, dists) {
                                        for i in 0..batch.num_rows() {
                                            vector_results.push((ids.value(i).to_string(), dists.value(i)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 路3：FTS5 全文检索 (BM25)
    let mut fts5_results: Vec<(String, f32)> = Vec::new();
    let conn = connect_sqlite(&data_dir)?;

    let fts5_query = fts5_search_query(&query);
    if !fts5_query.is_empty() {
        let sql = format!(
            "SELECT c.id, bm25(chunks_fts) as rank
             FROM chunks_fts
             JOIN chunks c ON chunks_fts.rowid = c.rowid
             WHERE chunks_fts MATCH '{}'
             ORDER BY rank
             LIMIT {}",
            fts5_query, settings.search_top_k * 4
        );

        if let Ok(mut stmt) = conn.prepare(&sql) {
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
            });
            if let Ok(rows) = rows {
                for row in rows.filter_map(Result::ok) {
                    fts5_results.push(row);
                }
            }
        }
    }

    // 路0：正则直接命中（如果用户提到了法条编号）
    let mut direct_results: Vec<(String, f32)> = Vec::new();
    if !article_numbers.is_empty() {
        let placeholders: String = article_numbers.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, 0.0 as score FROM chunks WHERE article_number IN ({}) LIMIT {}",
            placeholders, settings.search_top_k
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            let params: Vec<&dyn rusqlite::ToSql> = article_numbers.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            if let Ok(rows) = stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
            }) {
                for row in rows.filter_map(Result::ok) {
                    direct_results.push(row);
                }
            }
        }
    }

    println!(">>> [检索结果] 向量: {}, FTS5: {}, 正则直接: {}",
        vector_results.len(), fts5_results.len(), direct_results.len());

    // ========== RRF 融合 (Reciprocal Rank Fusion) ==========
    let mut rrf_scores: HashMap<String, f32> = HashMap::new();
    const RRF_K: f32 = 60.0;

    // 向量路
    for (i, (id, _dist)) in vector_results.iter().enumerate() {
        let score = 1.0 / (RRF_K + (i + 1) as f32);
        *rrf_scores.entry(id.clone()).or_insert(0.0) += score;
    }

    // FTS5 路
    for (i, (id, _rank)) in fts5_results.iter().enumerate() {
        let score = 1.0 / (RRF_K + (i + 1) as f32);
        *rrf_scores.entry(id.clone()).or_insert(0.0) += score;
    }

    // 正则直接路（最高权重）
    for (i, (id, _score)) in direct_results.iter().enumerate() {
        let score = 10.0 / (1.0 + (i + 1) as f32); // 高权重
        *rrf_scores.entry(id.clone()).or_insert(0.0) += score;
    }

    let mut fused_ids: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    fused_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let fused_ids_only: Vec<String> = fused_ids.iter().map(|(id, _)| id.clone()).take(100).collect();

    if fused_ids_only.is_empty() {
        println!(">>> [警告] 所有检索路都为空，返回空结果");
        return Ok(Vec::new());
    }

    // ========== Small-to-Big: 回表查询获取完整 Chunk 信息 + 完整法条 ==========
    let final_results = {
        let placeholders: String = fused_ids_only.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, content, law_name, category, region, publish_date, part, chapter, article_number
             FROM chunks WHERE id IN ({})",
            placeholders
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
        let params = rusqlite::params_from_iter(fused_ids_only.iter());
        let mut stmt_iter = stmt.query(params).map_err(|e| e.to_string())?;

        let mut chunk_map: HashMap<String, LawChunk> = HashMap::new();
        while let Some(row) = stmt_iter.next().map_err(|e| e.to_string())? {
            let id: String = row.get::<_, String>(0).map_err(|e| e.to_string())?;
            let law_name: String = row.get::<_, String>(2).map_err(|e| e.to_string())?;
            let article_number: String = row.get::<_, String>(8).map_err(|e| e.to_string())?;

            // Small-to-Big: 获取完整法条
            let full_article = fetch_full_article(&conn, &law_name, &article_number);

            chunk_map.insert(
                id.clone(),
                LawChunk {
                    id,
                    _distance: 0.0,
                    content: row.get::<_, String>(1).map_err(|e| e.to_string())?,
                    law_name: law_name.clone(),
                    category: row.get::<_, String>(3).map_err(|e| e.to_string())?,
                    region: row.get::<_, String>(4).map_err(|e| e.to_string())?,
                    publish_date: row.get::<_, String>(5).map_err(|e| e.to_string())?,
                    part: row.get::<_, String>(6).unwrap_or_default(),
                    chapter: row.get::<_, String>(7).unwrap_or_default(),
                    article_number,
                    source_file: format!("{}.txt", law_name),
                    full_article,
                },
            );
        }

        let mut results = Vec::new();
        for (id, rrf_score) in fused_ids.iter() {
            if let Some(mut chunk) = chunk_map.get(id).cloned() {
                chunk._distance = *rrf_score;

                let should_keep = if chunk.category != "地方法规" {
                    true
                } else {
                    if let Some(ref target_region) = filter_region {
                        chunk.region.contains(target_region)
                    } else {
                        false
                    }
                };

                if should_keep {
                    results.push(chunk);
                }
            }
        }
        results
    };

    // ========== Rerank 重排 (带降级) ==========
    let query_clone = query.clone();
    let chunks_clone = final_results.clone();
    let settings_clone = settings.clone();
    let search_top_k = settings.search_top_k;

    let reranked = tokio::task::spawn_blocking(move || {
        rerank_chunks(&query_clone, chunks_clone, &settings_clone)
    })
    .await
    .map_err(|e| e.to_string())?
    .unwrap_or_else(|_e: String| {
        println!(">>> [降级] Reranker 失败，使用 RRF 原始排序");
        final_results.into_iter().take(search_top_k * 2).collect()
    });

    println!(">>> [最终] 返回 {} 个结果", reranked.len());

    Ok(reranked
        .into_iter()
        .take(settings.search_top_k)
        .collect())
}

// ==========================================
// 5. Tauri 命令
// ==========================================

// 5.1 智能体搜索命令 (Agent)
#[tauri::command]
async fn start_agent_search(
    window: tauri::Window,
    query: String,
    event_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<LawChunk>, String> {
    let should_run = Arc::new(AtomicBool::new(true));
    {
        let mut flags = state.agent_abort_flags.lock().unwrap();
        flags.insert(event_id.clone(), should_run.clone());
    }

    macro_rules! check_abort {
        () => {
            if !should_run.load(Ordering::Relaxed) {
                // 清理并返回中断信号
                let mut flags = state.agent_abort_flags.lock().unwrap();
                flags.remove(&event_id);
                return Err("深度思考已手动停止".to_string());
            }
        };
    }

    let settings = state.settings.lock().unwrap().clone();
    let (model, base_url, api_key, max_loops) = if settings.use_external_chat_api {
        if settings.external_chat_api_choice == 2 {
            (
                settings.external_chat_model_2.clone(),
                settings.external_chat_base_url_2.clone(),
                settings.external_chat_api_key_2.clone(),
                settings.max_agent_loops,
            )
        } else {
            (
                settings.external_chat_model.clone(),
                settings.external_chat_base_url.clone(),
                settings.external_chat_api_key.clone(),
                settings.max_agent_loops,
            )
        }
    } else {
        (
            settings.chat_model.clone(),
            settings.chat_base_url.clone(),
            settings.chat_api_key.clone(),
            settings.max_agent_loops,
        )
    };
    let enable_thinking = settings.use_external_chat_api && settings.external_chat_api_choice == 2;

    let mut completed_log: Vec<CompletedTask> = vec![];

    // 使用 HashSet 收集 ID 去重，Vec 收集结果
    let mut all_found_chunks: Vec<LawChunk> = vec![];
    let mut seen_ids: HashSet<String> = HashSet::new();

    check_abort!();

    window
        .emit(
            "agent-update",
            AgentUpdateEvent {
                step_type: "planning".into(),
                todo_list: vec![],
                completed_log: vec![],
                current_task: None,
                thought: Some("正在拆解法律问题...".into()),
            },
        )
        .unwrap();

    let plan_prompt = PLANNER_PROMPT.replace("{user_query}", &query);
    println!(">>> Agent Planning...");
    let mut todo_list: Vec<String> = match call_llm(&model, &plan_prompt, &base_url, &api_key, enable_thinking).await
    {
        Ok(json) => {
            println!(">>> LLM Raw Output: {}", json);
            let clean = clean_json_str(&json);
            println!(">>> Cleaned JSON: {}", clean);
            match serde_json::from_str::<Vec<String>>(&clean) {
                Ok(list) => {
                    println!(">>> Parsed Task List: {:?}", list);
                    list
                }
                Err(e) => {
                    println!(">>> JSON Parse Error: {}", e);
                    // 如果解析失败，回退到原始查询
                    vec![query.clone()]
                }
            }
        }
        Err(_) => vec![query.clone()],
    };

    let mut loop_count = 0;
    let limit = if max_loops <= 0 { 99 } else { max_loops };

    while !todo_list.is_empty() && loop_count < limit {
        check_abort!();
        loop_count += 1;
        let current_task = todo_list.remove(0);
        println!(
            ">>> [Agent] Step {}: Executing task '{}'",
            loop_count, current_task
        );
        window
            .emit(
                "agent-update",
                AgentUpdateEvent {
                    step_type: "executing".into(),
                    todo_list: todo_list.clone(),
                    completed_log: completed_log.clone(),
                    current_task: Some(current_task.clone()),
                    thought: None,
                },
            )
            .unwrap();

        let search_res = search_law_logic(current_task.clone(), None, &state).await;

        check_abort!();

        let mut result_text = String::new();
        let mut found_count = 0;
        let step_max_chunks = 10; 

        match search_res {
            Ok(chunks) => {
                for r in chunks {
                    // 1.2 阈值过滤
                    if r._distance < 1.2 {
                        if found_count >= step_max_chunks {
                            break;
                        }
                        found_count += 1;
                        // 收集文本给 Agent 看
                        result_text.push_str(&format!(
                            "法规：《{}》{}\n内容：{}\n\n",
                            r.law_name, r.article_number, r.content
                        ));

                        // 收集对象给前端
                        if !seen_ids.contains(&r.id) {
                            seen_ids.insert(r.id.clone());
                            all_found_chunks.push(r);
                        }
                    }
                }
            }
            Err(e) => {
                result_text = format!("搜索出错: {}", e);
            }
        }

        if result_text.trim().is_empty() {
            result_text = "未找到直接相关法条。".to_string();
            println!(">>> [Agent] No results found for this task.");
        } else {
            println!(">>> [Agent] Found {} relevant chunks.", found_count);
        }
        check_abort!();
        window
            .emit(
                "agent-update",
                AgentUpdateEvent {
                    step_type: "thinking".into(),
                    todo_list: todo_list.clone(),
                    completed_log: completed_log.clone(),
                    current_task: Some(current_task.clone()),
                    thought: Some("正在评估检索结果...".into()),
                },
            )
            .unwrap();

        let review_prompt = EXECUTOR_PROMPT
            .replace("{user_query}", &query)
            .replace("{current_task}", &current_task)
            .replace("{search_results}", &result_text)
            .replace(
                "{remaining_todo_list}",
                &serde_json::to_string(&todo_list).unwrap_or("[]".into()),
            );
        check_abort!();
        match call_llm(&model, &review_prompt, &base_url, &api_key, enable_thinking).await {
            Ok(json) => {
                if let Some(res) = parse_executor_response(&json) {
                    println!(">>> [Agent] Thought: {}", res.thought);
                    if !res.new_todo_list.is_empty() {
                        println!(">>> [Agent] Updated List: {:?}", res.new_todo_list);
                        todo_list = res.new_todo_list;
                    }
                    completed_log.push(CompletedTask {
                        task: current_task,
                        thought: res.thought,
                    });
                } else {
                    println!(">>> [Agent] Response Parse Failed: {}", json);
                    completed_log.push(CompletedTask {
                        task: current_task,
                        thought: "解析思考结果失败，继续执行原计划。".into(),
                    });
                }
            }
            Err(e) => {
                println!(">>> [Agent] LLM Reflection Error: {}", e);
                completed_log.push(CompletedTask {
                    task: current_task,
                    thought: "LLM 调用失败，跳过此步分析。".into(),
                });
            }
        }
    }

    {
        let mut flags = state.agent_abort_flags.lock().unwrap();
        flags.remove(&event_id);
    }

    window
        .emit(
            "agent-update",
            AgentUpdateEvent {
                step_type: "finished".into(),
                todo_list: vec![],
                completed_log: completed_log,
                current_task: None,
                thought: Some("所有任务执行完毕，正在生成最终回答...".into()),
            },
        )
        .unwrap();
    println!(
        ">>> [Agent] Finished. Total chunks found: {}",
        all_found_chunks.len()
    );
    Ok(all_found_chunks)
}

// 5.2 普通搜索命令 (Search)
#[tauri::command]
async fn search_law(
    query: String,
    filter_region: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<LawChunk>, String> {
    search_law_logic(query, filter_region, &state).await
}

// 5.3 其他命令 (Others)
#[tauri::command]
fn check_db_status(state: tauri::State<'_, AppState>) -> bool {
    let data_dir = get_effective_data_dir(&state);
    let lancedb_path = data_dir.join("law_db.lancedb");
    lancedb_path.exists()
}

#[tauri::command]
fn add_draft_material(chunk: LawChunk, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "INSERT INTO draft_materials (law_id, law_name, article_number, content) VALUES (?1, ?2, ?3, ?4) ON CONFLICT(law_id) DO NOTHING",
        rusqlite::params![chunk.id, chunk.law_name, chunk.article_number, chunk.content],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn get_draft_materials(state: tauri::State<'_, AppState>) -> Result<Vec<DraftMaterial>, String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let mut stmt = conn.prepare("SELECT id, law_id, law_name, article_number, content, added_at FROM draft_materials ORDER BY added_at DESC").map_err(|e| e.to_string())?;
    let items = stmt
        .query_map([], |row| {
            Ok(DraftMaterial {
                id: row.get(0)?,
                law_id: row.get(1)?,
                law_name: row.get(2)?,
                article_number: row.get(3)?,
                content: row.get(4)?,
                added_at: row.get(5)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    Ok(items)
}

#[tauri::command]
fn remove_draft_material(law_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "DELETE FROM draft_materials WHERE law_id = ?1",
        rusqlite::params![law_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn clear_draft_materials(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute("DELETE FROM draft_materials", [])
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn add_template(
    name: String,
    content: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute("INSERT INTO custom_templates (name, content) VALUES (?1, ?2) ON CONFLICT(name) DO UPDATE SET content = excluded.content", rusqlite::params![name, content]).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn get_templates(state: tauri::State<'_, AppState>) -> Result<Vec<CustomTemplate>, String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let mut stmt = conn
        .prepare("SELECT id, name, content FROM custom_templates ORDER BY id DESC")
        .map_err(|e| e.to_string())?;
    let items = stmt
        .query_map([], |row| {
            Ok(CustomTemplate {
                id: row.get(0)?,
                name: row.get(1)?,
                content: row.get(2)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    Ok(items)
}

#[tauri::command]
fn delete_template(id: i32, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "DELETE FROM custom_templates WHERE id = ?1",
        rusqlite::params![id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn search_law_by_name(
    query: String,
    limit: usize,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<LawNameSuggestion>, String> {
    let data_dir = get_effective_data_dir(&state);
    let conn = connect_sqlite(&data_dir)?;

    let sql = "SELECT DISTINCT law_name, region, category FROM full_texts WHERE law_name LIKE ? LIMIT 200";
    let query_pattern = format!("%{}%", query);

    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;

    let mut suggestions: Vec<LawNameSuggestion> = stmt
        .query_map(rusqlite::params![query_pattern], |row| {
            Ok(LawNameSuggestion {
                name: row.get(0)?,
                region: row.get(1)?,
                category: row.get(2)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();

    fn get_category_priority(cat: &str) -> i32 {
        match cat {
            "法律" => 1,
            "司法解释" => 2,
            "行政法规" => 3,
            "地方法规" => 4,
            _ => 99,
        }
    }

    suggestions.sort_by(|a, b| {
        let p_a = get_category_priority(&a.category);
        let p_b = get_category_priority(&b.category);

        if p_a != p_b {
            p_a.cmp(&p_b)
        } else {
            a.name.len().cmp(&b.name.len())
        }
    });

    if suggestions.len() > limit {
        suggestions.truncate(limit);
    }

    Ok(suggestions)
}

#[tauri::command]
fn get_article_snippet(
    law_name_query: Option<String>,
    article_number: String,
    current_law_name: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let data_dir = get_effective_data_dir(&state);
    let conn = connect_sqlite(&data_dir)?;

    let target_law = match law_name_query {
        Some(name) => name,
        None => current_law_name,
    };

    let sql = "SELECT content FROM chunks WHERE law_name LIKE ? AND article_number = ? LIMIT 1";
    let law_pattern = format!("%{}%", target_law);

    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mut rows = stmt
        .query(rusqlite::params![law_pattern, article_number])
        .map_err(|e| e.to_string())?;

    if let Some(row) = rows.next().map_err(|e| e.to_string())? {
        Ok(row.get(0).map_err(|e| e.to_string())?)
    } else {
        Ok(format!("未找到《{}》的{}", target_law, article_number))
    }
}

#[tauri::command]
async fn check_ai_connection(
    base_url: String,
    api_key: String,
    model: String,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/models", base_url.trim_end_matches('/'));

    let res = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .map_err(|e| format!("连接失败: 网络请求错误 ({})", e))?;

    if !res.status().is_success() {
        return Err(format!("连接失败: 服务器返回状态码 {}", res.status()));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| format!("解析失败: {}", e))?;

    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
        let model_exists = data
            .iter()
            .any(|m| m.get("id").and_then(|id| id.as_str()) == Some(&model));

        if model_exists {
            Ok(format!("连接成功！发现模型: {}", model))
        } else {
            Ok(format!(
                "连接通畅，但在列表中未找到模型 '{}' (可能仍可用)",
                model
            ))
        }
    } else {
        Ok("连接成功！(未能验证模型名称)".to_string())
    }
}

#[tauri::command]
fn get_full_text(source_file: String, state: tauri::State<'_, AppState>) -> Result<String, String> {
    let data_dir = get_effective_data_dir(&state);
    let conn = connect_sqlite(&data_dir)?;
    let raw_name = source_file.trim_end_matches(".txt");

    let mut stmt = conn
        .prepare("SELECT full_text FROM full_texts WHERE law_name = ? LIMIT 1")
        .map_err(|e| e.to_string())?;

    let mut rows = stmt
        .query(rusqlite::params![raw_name])
        .map_err(|e| e.to_string())?;

    if let Some(row) = rows.next().map_err(|e| e.to_string())? {
        return Ok(row.get(0).map_err(|e| e.to_string())?);
    }

    let fuzzy_pattern = format!("%{}", raw_name);

    let mut stmt = conn.prepare(
        "SELECT full_text FROM full_texts WHERE law_name LIKE ? ORDER BY length(law_name) ASC LIMIT 1"
    ).map_err(|e| e.to_string())?;

    let mut rows = stmt
        .query(rusqlite::params![fuzzy_pattern])
        .map_err(|e| e.to_string())?;

    if let Some(row) = rows.next().map_err(|e| e.to_string())? {
        return Ok(row.get(0).map_err(|e| e.to_string())?);
    }

    let loose_pattern = format!("%{}%", raw_name);
    let mut stmt = conn.prepare(
        "SELECT full_text FROM full_texts WHERE law_name LIKE ? ORDER BY length(law_name) ASC LIMIT 1"
    ).map_err(|e| e.to_string())?;

    let mut rows = stmt
        .query(rusqlite::params![loose_pattern])
        .map_err(|e| e.to_string())?;

    if let Some(row) = rows.next().map_err(|e| e.to_string())? {
        return Ok(row.get(0).map_err(|e| e.to_string())?);
    }

    Err(format!("未找到法律文件：{}", raw_name))
}

#[tauri::command]
async fn chat_stream(
    app: AppHandle,
    query: String,
    context_chunks: Vec<String>,
    mode: String,
    event_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let settings = state.settings.lock().unwrap().clone();
    println!("[ChatStream] 调用: mode={}, use_external={}, external_choice={}", 
        mode, settings.use_external_chat_api, settings.external_chat_api_choice);

    // 深度模式下，允许更多的上下文进入（例如 Top 10），普通模式 Top 5
    let limit = if mode == "deep" || mode == "draft" {
        settings.chat_top_k * 2
    } else {
        settings.chat_top_k
    };

    let selected_chunks = if context_chunks.len() > limit {
        &context_chunks[..limit]
    } else {
        &context_chunks[..]
    };

    let context_str = selected_chunks.join("\n\n");

    // === 分析 Prompts ===

    // 1. 深度思考模式 Prompt：专业法律意见书风格
    let deep_prompt = format!(
        r#"你是一位资深的中国法律顾问。用户提出了一个具体的法律问题，你已经通过检索工具找到了相关的法律条文。
你的任务是根据这些法条，为用户撰写一份专业的《法律检索分析报告》。

要求：
1. 每个结论必须引用具体法条（格式：《XX法》第X条）
2. 如果检索结果不足，明确说明缺少的部分
3. 专业但通俗，避免过度术语堆砌
4. 不编造法条，不做绝对承诺
5. 不需要寒暄

输出结构：

一、核心结论
用一句话回答用户的核心问题。

二、法律依据分析
针对争议点逐条分析：
- 法条依据：《XX法》第X条规定...
- 适用分析：对用户情况的具体解读
- 注意事项：适用条件或例外情况

三、实操建议
1. 证据准备：需要保留哪些材料
2. 维权路径：协商/仲裁/诉讼的具体步骤
3. 时间节点：诉讼时效、关键期限

---
【检索到的法条上下文】：
{}
"#,
        context_str
    );

    // 2. 普通模式 Prompt
    let simple_prompt = format!(
        r#"你是一个法条检索助手。请基于以下检索结果，先简要评估其与用户问题的相关性。然后再给出回答。不需要寒暄。

【检索到的法条】：
{}

要求：
1. 如果法条和问题高度相关，请直接根据法条内容回答用户问题，答案简洁明了，需要引用具体相关法条。不相关法条请予以忽略。
输出示例：
```
关于（用户问题）的问题，（基于xx法xx条，此行为可能构成……）
```
2. 如果法条不相关，请直接告知用户“未找到直接相关依据”，并建议更换搜索词。搜索词应基于法条相似度Embedding的方向设计。
输出示例：
```
查找到的法条相关度较低，根据您的问题，建议以下搜索词重新搜索：（数个搜索词）
```
3. 如果法条相关度完全不足，请告知用户检查向量模型和数据库是否匹配。
"#,
        context_str
    );

    let draft_prompt = format!(
        r#"你是一位专业的法律文书起草专家。用户提供了一些参考法条和具体的写作要求。
你的任务是根据这些素材，起草一份高质量的法律文书或段落。

【参考法条/素材】：
{}

【要求】：
1. 格式规范，用词严谨。
2. 必须充分利用提供的素材中的法律依据。
3. 如果用户提供了模版，请严格遵循模版的结构。
4. 直接输出文书正文。
5. 不要任何寒暄。
6. 不要使用超过提供法条之外的法条文本。
"#,
        context_str
    );

    // 根据 mode 选择 prompt
    let system_prompt = match mode.as_str() {
        "deep" => deep_prompt,
        "draft" => draft_prompt,
        _ => simple_prompt,
    };

    let user_prompt = if mode == "draft" {
        format!("【写作指令】：{}\n\n请开始起草：", query)
    } else {
        format!("用户问题：{}\n\n请开始分析：", query)
    };
    let event_id_for_task = event_id.clone();

    let (chat_base_url, chat_api_key, chat_model) = if settings.use_external_chat_api {
        match settings.external_chat_api_choice {
            2 => (
                settings.external_chat_base_url_2.clone(),
                settings.external_chat_api_key_2.clone(),
                settings.external_chat_model_2.clone(),
            ),
            _ => (
                settings.external_chat_base_url.clone(),
                settings.external_chat_api_key.clone(),
                settings.external_chat_model.clone(),
            ),
        }
    } else {
        (
            settings.chat_base_url.clone(),
            settings.chat_api_key.clone(),
            settings.chat_model.clone(),
        )
    };

    let chat_task = tauri::async_runtime::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap();
        let url = format!(
            "{}/chat/completions",
            chat_base_url.trim_end_matches('/')
        );
        println!("[ChatStream] 发送请求到: {}, 模型: {}", url, chat_model);

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", chat_api_key))
            .json(&serde_json::json!({
                "model": chat_model,
                "messages": [
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": user_prompt }
                ],
                "stream": true,
                "max_tokens": if mode == "deep" { 4096 } else { 2048 },
                "temperature": if mode == "deep" { 0.4 } else { 0.3 }
            }))
            .send()
            .await;

        match response {
            Ok(res) => {
                let mut stream = res.bytes_stream();
                let mut line_buf = String::new();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            line_buf.push_str(&text);

                            while let Some(newline_pos) = line_buf.find('\n') {
                                let line = line_buf[..newline_pos].to_string();
                                line_buf = line_buf[newline_pos + 1..].to_string();

                                let trimmed_line = line.trim();
                                // 兼容 data:[DONE] 和 data: [DONE]
                                if trimmed_line == "[DONE]"
                                    || trimmed_line == "data:[DONE]"
                                    || trimmed_line == "data: [DONE]"
                                {
                                    let _ = app.emit(&event_id_for_task, "[DONE]");
                                    return;
                                } else if trimmed_line.starts_with("data:") && trimmed_line.len() > 5 {
                                    let json_str = trimmed_line[5..].trim_start();
                                    if json_str.is_empty() {
                                        continue;
                                    }
                                    if let Ok(json) =
                                        serde_json::from_str::<serde_json::Value>(json_str)
                                    {
                                        if let Some(reasoning) =
                                            json["choices"][0]["delta"]["reasoning_content"].as_str()
                                        {
                                            let converted = reasoning
                                                .replace("<longcat_think>", "<think>")
                                                .replace("</longcat_think>", "</think>");
                                            let wrapped = format!(
                                                "<think>{}</think>",
                                                converted.replace("<think>", "").replace("</think>", "")
                                            );
                                            let _ = app.emit(&event_id_for_task, &wrapped);
                                        }

                                        if let Some(content) =
                                            json["choices"][0]["delta"]["content"].as_str()
                                        {
                                            let clean = content
                                                .replace("<think>", "")
                                                .replace("</think>", "")
                                                .replace("<longcat_think>", "")
                                                .replace("</longcat_think>", "");
                                            let _ = app.emit(&event_id_for_task, &clean);
                                        } else if let Some(content) =
                                            json["message"]["content"].as_str()
                                        {
                                            let clean = content
                                                .replace("<think>", "")
                                                .replace("</think>", "")
                                                .replace("<longcat_think>", "")
                                                .replace("</longcat_think>", "");
                                            let _ = app.emit(&event_id_for_task, &clean);
                                        } else if let Some(content) = json["content"].as_str() {
                                            let clean = content
                                                .replace("<think>", "")
                                                .replace("</think>", "")
                                                .replace("<longcat_think>", "")
                                                .replace("</longcat_think>", "");
                                            let _ = app.emit(&event_id_for_task, &clean);
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            println!("[ChatStream] Stream error: {}", e);
                            let _ = app.emit(&event_id_for_task, format!("[Error: Stream error: {}]", e));
                        }
                    }
                }
                let _ = app.emit(&event_id_for_task, "[DONE]");
            }
            Err(e) => {
                println!("[ChatStream] Request error: {}", e);
                let _ = app.emit(&event_id_for_task, format!("[Error: Request failed: {}]", e));
            }
        }
    });

    // 3. 将任务句柄存入 Map (使用原始的 event_id)
    {
        let mut tasks = state.chat_tasks.lock().unwrap();
        tasks.insert(event_id, chat_task);
    }

    Ok(())
}

#[tauri::command]
fn stop_chat(event_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut tasks = state.chat_tasks.lock().unwrap();
    if let Some(handle) = tasks.remove(&event_id) {
        handle.abort(); // 强制中止任务
        println!(">>> Chat task aborted: {}", event_id);
    }
    Ok(())
}

#[tauri::command]
fn stop_task(event_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    // 1. 尝试停止 Chat Stream 任务
    let mut tasks = state.chat_tasks.lock().unwrap();
    if let Some(handle) = tasks.remove(&event_id) {
        handle.abort();
        println!(">>> Chat task aborted: {}", event_id);
    }

    // 2. 尝试停止 Agent 循环
    let mut flags = state.agent_abort_flags.lock().unwrap();
    if let Some(flag) = flags.remove(&event_id) {
        flag.store(false, Ordering::Relaxed); // 设置开关为 false
        println!(">>> Agent loop abort signaled: {}", event_id);
    }

    Ok(())
}

#[tauri::command]
fn get_settings(state: tauri::State<'_, AppState>) -> AppSettings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
fn save_settings(
    new_settings: AppSettings,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let mut guard = state.settings.lock().unwrap();
    *guard = new_settings.clone();

    let json = serde_json::to_string_pretty(&new_settings).map_err(|e| e.to_string())?;
    let _ = fs::write(&state.settings_path, json);

    Ok(())
}

// === User Data CRUD Commands ===

#[tauri::command]
fn add_favorite(
    chunk: LawChunk,
    folder_id: Option<i32>, // 修改：接收 folder_id
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    // 使用 REPLACE INTO 或者 ON CONFLICT 更新 folder_id
    conn.execute(
        "INSERT INTO favorites (law_id, law_name, article_number, content, folder_id) 
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(law_id) DO UPDATE SET folder_id = excluded.folder_id",
        rusqlite::params![
            chunk.id,
            chunk.law_name,
            chunk.article_number,
            chunk.content,
            folder_id
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn move_favorite(
    law_id: String,
    folder_id: Option<i32>,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "UPDATE favorites SET folder_id = ?2 WHERE law_id = ?1",
        rusqlite::params![law_id, folder_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn remove_favorite(law_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "DELETE FROM favorites WHERE law_id = ?1",
        rusqlite::params![law_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn create_folder(name: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "INSERT INTO favorite_folders (name) VALUES (?1)",
        rusqlite::params![name],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn get_folders(state: tauri::State<'_, AppState>) -> Result<Vec<UserFolder>, String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let mut stmt = conn
        .prepare("SELECT id, name, created_at FROM favorite_folders ORDER BY created_at ASC")
        .map_err(|e| e.to_string())?;

    let folders = stmt
        .query_map([], |row| {
            Ok(UserFolder {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();

    Ok(folders)
}

#[tauri::command]
fn delete_folder(folder_id: i32, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute(
        "DELETE FROM favorites WHERE folder_id = ?1",
        rusqlite::params![folder_id],
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "DELETE FROM favorite_folders WHERE id = ?1",
        rusqlite::params![folder_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn get_favorites(state: tauri::State<'_, AppState>) -> Result<Vec<UserFavorite>, String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let mut stmt = conn.prepare("SELECT id, law_id, law_name, article_number, content, created_at, tags, folder_id FROM favorites ORDER BY created_at DESC")
        .map_err(|e| e.to_string())?;

    let favorites = stmt
        .query_map([], |row| {
            Ok(UserFavorite {
                id: row.get(0)?,
                law_id: row.get(1)?,
                law_name: row.get(2)?,
                article_number: row.get(3)?,
                content: row.get(4)?,
                created_at: row.get(5)?,
                tags: row.get(6)?,
                folder_id: row.get(7)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();

    Ok(favorites)
}

#[tauri::command]
fn check_is_favorite(law_id: String, state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let count: i32 = conn
        .query_row(
            "SELECT count(*) FROM favorites WHERE law_id = ?1",
            rusqlite::params![law_id],
            |row| row.get(0),
        )
        .unwrap_or(0);
    Ok(count > 0)
}

#[tauri::command]
fn add_history(query: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "REPLACE INTO search_history (query, timestamp) VALUES (?1, ?2)",
        rusqlite::params![query, timestamp],
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "DELETE FROM search_history WHERE id NOT IN (SELECT id FROM search_history ORDER BY timestamp DESC LIMIT 50)",
        [],
    ).map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
fn get_history(state: tauri::State<'_, AppState>) -> Result<Vec<String>, String> {
    let conn = connect_user_db(&state.user_db_path)?;
    let mut stmt = conn
        .prepare("SELECT query FROM search_history ORDER BY timestamp DESC")
        .map_err(|e| e.to_string())?;

    let history = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    Ok(history)
}

#[tauri::command]
fn clear_history(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let conn = connect_user_db(&state.user_db_path)?;
    conn.execute("DELETE FROM search_history", [])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ==========================================
// 6. 程序入口
// ==========================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // 1. 获取 exe 目录 (便携模式检测)
            let mut exe_path = std::env::current_exe()?;
            exe_path.pop();
            let portable_settings = exe_path.join("settings.json");
            let portable_user_db = exe_path.join("user_data.db");

            // 2. 获取系统 AppData 目录
            let app_config_dir = app.path().resolve("", BaseDirectory::AppConfig)?;
            if !app_config_dir.exists() {
                std::fs::create_dir_all(&app_config_dir)?;
            }
            let system_settings = app_config_dir.join("settings.json");
            let system_user_db = app_config_dir.join("user_data.db");

            // 3. 决策路径
            // 规则：如果 exe 旁边有配置文件，就认为是便携模式，数据库也读旁边的
            // 否则全部走系统目录
            let (final_settings_path, final_user_db_path) = if portable_settings.exists() {
                println!(">>> Mode: Portable");
                (portable_settings, portable_user_db)
            } else {
                println!(">>> Mode: Standard (AppData)");
                (system_settings, system_user_db)
            };

            // 4. 加载配置
            let settings = if final_settings_path.exists() {
                load_settings_from_disk(&final_settings_path)
            } else {
                println!(">>> Creating default settings at {:?}", final_settings_path);
                let default = AppSettings::default();
                // 首次运行自动生成配置文件
                let json = serde_json::to_string_pretty(&default)?;
                let _ = fs::write(&final_settings_path, json);
                default
            };

            // 5. 初始化用户数据库
            // 如果文件不存在，connect_user_db 内部会自动创建
            let _ = connect_user_db(&final_user_db_path).map_err(|e| {
                eprintln!("User DB init failed: {}", e);
                e
            });

            // 6. 默认资源路径 (content.db)
            // 同样支持便携优先: exe/data > resource/app_data
            let portable_data_dir = exe_path.join("data");
            let resource_data_dir = app
                .path()
                .resolve("resources/app_data", BaseDirectory::Resource)?;

            let final_app_data_dir = if portable_data_dir.exists() {
                portable_data_dir
            } else {
                resource_data_dir
            };

            app.manage(AppState {
                settings: Mutex::new(settings),
                settings_path: final_settings_path,
                app_data_dir: final_app_data_dir,
                user_db_path: final_user_db_path,
                chat_tasks: Mutex::new(HashMap::new()),
                agent_abort_flags: Mutex::new(HashMap::new()),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            search_law,
            chat_stream,
            stop_chat,
            stop_task,
            get_settings,
            save_settings,
            search_law_by_name,
            get_full_text,
            check_ai_connection,
            get_article_snippet,
            check_db_status,
            start_agent_search,
            // User Data Commands
            add_favorite,
            remove_favorite,
            get_favorites,
            check_is_favorite,
            add_history,
            get_history,
            clear_history,
            create_folder,
            get_folders,
            delete_folder,
            move_favorite,
            add_draft_material,
            get_draft_materials,
            remove_draft_material,
            clear_draft_materials,
            add_template,
            get_templates,
            delete_template
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
