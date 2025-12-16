# LLM CLI to OpenAI API

Personal OpenAI-compatible API server wrapping **Codex CLI** and **Claude CLI**.
Use your existing CLI subscriptions with any OpenAI SDK client (Continue, Cursor, LangChain, etc).

个人用 OpenAI 兼容 API 服务，封装 Codex CLI 和 Claude CLI，让你的 CLI 订阅可以配合任何 OpenAI SDK 客户端使用。

## Features

- **OpenAI-compatible endpoints**: `/v1/completions`, `/v1/chat/completions`
- **Auto-routing**: Model name prefix determines backend (Claude or Codex)
- **System prompt support**: Extracts `system` messages and passes to `--system-prompt`
- **Smart retry**: 30 retries with exponential backoff when queue is full
- **Client disconnect detection**: Terminates subprocess when client disconnects
- **Concurrency control**: Semaphore-based request queueing
- **Bearer auth**: Simple token authentication
- **Cross-platform**: Windows, macOS, Linux

## Supported Models

| Model Prefix | Backend | Example |
|--------------|---------|---------|
| `claude-haiku-*` | Claude CLI | `claude-haiku-4-5-20251001` |
| `claude-sonnet-*` | Claude CLI | `claude-sonnet-4-20251001` |
| `claude-opus-*` | Claude CLI | `claude-opus-4-20251001` |
| Other | Codex CLI | `gpt-5.1`, `o3-mini` |

## Quick Start

### 1. Install dependencies

```bash
# Create environment (Python 3.11+)
conda create -n llm-api python=3.13
conda activate llm-api

pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and edit:

```bash
# CLI paths (auto-detected if on PATH)
CODEX_BIN=C:\Users\xxxx\AppData\Roaming\npm\codex.cmd
CLAUDE_BIN=C:\Users\xxxx\AppData\Roaming\npm\claude.cmd

# Required for Claude CLI on Windows
CLAUDE_CODE_GIT_BASH_PATH=C:\Program Files\Git\bin\bash.exe

# Server settings
API_AUTH_TOKEN=your-secret-token
MAX_CONCURRENT_RUNS=2
MAX_QUEUE_SIZE=249
API_HOST=127.0.0.1
API_PORT=8030

# Retry settings (queue full)
MAX_RETRY_ATTEMPTS=30
RETRY_BASE_DELAY=0.5
RETRY_MAX_DELAY=30.0
```

### 3. Run

```bash
python run.py
# Server starts at http://127.0.0.1:8030
```

## Usage Examples

### Health Check

```bash
curl http://127.0.0.1:8030/health
```

### Chat Completions with System Prompt

```bash
curl -X POST http://127.0.0.1:8030/v1/chat/completions \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20251001",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ]
  }'
```

### Windows CMD

```cmd
set API_AUTH_TOKEN=your-secret-token
curl -X POST http://127.0.0.1:8030/v1/chat/completions ^
  -H "Authorization: Bearer %API_AUTH_TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gpt-5.1\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

## Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8030/v1",
    api_key="your-secret-token"
)

# Use Codex backend
response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[{"role": "user", "content": "Hello"}]
)

# Use Claude backend with system prompt
response = client.chat.completions.create(
    model="claude-sonnet-4-20251001",
    messages=[
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Hello"}
    ]
)
```

## Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Liveness check |
| `POST /run` | No | Direct CLI invocation (model required) |
| `POST /v1/completions` | Bearer | OpenAI-compatible text completions |
| `POST /v1/chat/completions` | Bearer | OpenAI-compatible chat completions |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_BIN` | auto-detect | Path to Codex CLI |
| `CLAUDE_BIN` | auto-detect | Path to Claude CLI |
| `CLAUDE_CODE_GIT_BASH_PATH` | - | Git Bash path (required for Claude CLI on Windows) |
| `API_AUTH_TOKEN` | - | Bearer token for `/v1/*` endpoints |
| `MAX_CONCURRENT_RUNS` | 2 | Max parallel CLI processes |
| `MAX_QUEUE_SIZE` | max(4, 2×concurrent) | Request queue limit |
| `MAX_RETRY_ATTEMPTS` | 30 | Retries when queue is full |
| `RETRY_BASE_DELAY` | 0.5 | Base delay in seconds |
| `RETRY_MAX_DELAY` | 30.0 | Max delay cap in seconds |
| `API_HOST` | 127.0.0.1 | Server bind address |
| `API_PORT` | 8000 | Server port |

## Retry Mechanism

When the queue is full, requests retry with exponential backoff:

```
Retry 1: 0.5s  + jitter
Retry 2: 1.0s  + jitter
Retry 3: 2.0s  + jitter
Retry 4: 4.0s  + jitter
Retry 5: 8.0s  + jitter
Retry 6: 16.0s + jitter
Retry 7+: 30.0s (capped) + jitter
```

After 30 failed retries, returns HTTP 429.

## Client Disconnect Handling

- Detects when client disconnects during request processing
- Terminates running subprocess to free resources
- Returns HTTP 499 (Client Closed Request)

## Message Parsing

System messages are extracted and passed separately:

**Input:**
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Hello"}
  ]
}
```

**Claude CLI receives:**
- `--system-prompt "You are helpful. | Be concise."`
- `-p "user: Hello"`

**Codex CLI receives (via stdin):**
- `System: You are helpful. | Be concise. | user: Hello`

## Architecture

```
Request → FastAPI → Model Router → [Codex CLI | Claude CLI] → Response
                         ↓
              Semaphore (concurrency control)
                         ↓
              Queue + Retry (with exponential backoff)
                         ↓
              Client disconnect detection
```

## Load Testing

```bash
python test_concurrency.py --count 8 --endpoint v1/chat/completions \
  --base-url http://127.0.0.1:8030 --token your-secret-token
```

## Prerequisites

- Python 3.11+
- [Codex CLI](https://github.com/openai/codex) installed and authenticated
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) installed and authenticated

## License

MIT

## Related Projects

- [Codex CLI](https://github.com/openai/codex) - OpenAI's coding assistant CLI
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) - Anthropic's Claude CLI

---

**Note**: This service shells out to CLI tools. Ensure you comply with each tool's license and terms of service. Not intended for public deployment without proper security measures.
