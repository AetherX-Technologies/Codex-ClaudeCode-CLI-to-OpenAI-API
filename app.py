import asyncio
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

# Switch Windows to Selector policy to avoid Proactor WinError 10014 when forking
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Codex CLI Wrapper", version="0.1.0")

# Prefer env var; otherwise try auto-detect codex/codex.cmd.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("codex-api")

load_dotenv(override=False)

# Auto-detect Codex CLI
if os.name == "nt":
    _default_codex = shutil.which("codex.cmd") or shutil.which("codex")
else:
    _default_codex = shutil.which("codex")
CODEX_BIN = os.getenv("CODEX_BIN") or _default_codex

# Auto-detect Claude CLI
if os.name == "nt":
    _default_claude = shutil.which("claude.cmd") or shutil.which("claude")
else:
    _default_claude = shutil.which("claude")
CLAUDE_BIN = os.getenv("CLAUDE_BIN") or _default_claude

# Model routing: prefixes that route to Claude CLI
CLAUDE_MODEL_PREFIXES = ("claude-haiku-", "claude-sonnet-", "claude-opus-")
try:
    MAX_CONCURRENT_RUNS = max(1, int(os.getenv("MAX_CONCURRENT_RUNS", "2")))
except ValueError:
    MAX_CONCURRENT_RUNS = 2
EXEC_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
try:
    _queue_default = max(4, 2 * MAX_CONCURRENT_RUNS)
    MAX_QUEUE_SIZE = max(0, int(os.getenv("MAX_QUEUE_SIZE", str(_queue_default))))
except ValueError:
    MAX_QUEUE_SIZE = max(4, 2 * MAX_CONCURRENT_RUNS)

# 重试配置
try:
    MAX_RETRY_ATTEMPTS = max(1, int(os.getenv("MAX_RETRY_ATTEMPTS", "30")))
except ValueError:
    MAX_RETRY_ATTEMPTS = 30

try:
    RETRY_BASE_DELAY = max(0.1, float(os.getenv("RETRY_BASE_DELAY", "0.5")))
except ValueError:
    RETRY_BASE_DELAY = 0.5

try:
    RETRY_MAX_DELAY = max(1.0, float(os.getenv("RETRY_MAX_DELAY", "30.0")))
except ValueError:
    RETRY_MAX_DELAY = 30.0

_queue_lock = asyncio.Lock()
_queued_waiters = 0
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
logger.info(
    "config loaded: CODEX_BIN=%s CLAUDE_BIN=%s MAX_CONCURRENT_RUNS=%s MAX_QUEUE_SIZE=%s MAX_RETRY_ATTEMPTS=%s API_AUTH_TOKEN=%s",
    CODEX_BIN,
    CLAUDE_BIN,
    MAX_CONCURRENT_RUNS,
    MAX_QUEUE_SIZE,
    MAX_RETRY_ATTEMPTS,
    "set" if API_AUTH_TOKEN else "unset",
)


def _is_claude_model(model: str) -> bool:
    """Check if model should use Claude CLI based on prefix."""
    return model.lower().startswith(CLAUDE_MODEL_PREFIXES)

class RunRequest(BaseModel):
    prompt: str = Field(..., description='Prompt forwarded to LLM CLI via stdin')
    model: str = Field(..., description='Model name (required, e.g. gpt-5.1 or claude-sonnet-4-20251001)')
    args: List[str] = Field(default_factory=list, description='Extra CLI args, e.g. --temperature 0.2')
    timeout: Optional[float] = Field(None, description='Timeout in seconds; None disables it')
    clean: bool = Field(True, description='If true, return only cleaned stdout as output')

class RunResponse(BaseModel):
    output: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = Field(None, description="Optional timeout in seconds")


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = Field(None, description="Optional timeout in seconds")


class ChoiceMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[dict] = None
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


async def _require_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_AUTH_TOKEN:
        raise HTTPException(status_code=500, detail="API_AUTH_TOKEN is not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def _build_chat_prompt(messages: List[ChatMessage]) -> tuple[str, Optional[str]]:
    """Build prompt from chat messages, extracting system messages separately.

    Returns:
        tuple: (prompt, system_prompt)
        - prompt: Formatted conversation (user/assistant messages)
        - system_prompt: All system messages combined with | (or None if no system messages)
    """
    system_parts = []
    conversation_parts = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            # user, assistant, tool, function 等
            conversation_parts.append(f"{msg.role}: {msg.content}")

    # 用 | 替代换行符，避免 Windows subprocess 传参问题
    prompt = " | ".join(conversation_parts)
    system_prompt = " | ".join(system_parts) if system_parts else None

    return prompt, system_prompt


def _token_count(text: str) -> int:
    return len(text.split())


class ClientDisconnectedError(Exception):
    """Raised when client disconnects during request processing."""
    pass


async def _execute_llm(
    prompt: str,
    model: str,
    args: List[str],
    timeout: Optional[float],
    clean: bool = True,
    request: Optional[Request] = None,
    system_prompt: Optional[str] = None,
):
    """Execute LLM CLI (Codex or Claude) based on model name.

    Supports client disconnect detection - if client disconnects, the subprocess
    will be terminated and resources released.

    Args:
        system_prompt: Custom system prompt (for Claude CLI). If None, uses default.
    """
    use_claude = _is_claude_model(model)

    if use_claude:
        if not CLAUDE_BIN:
            raise HTTPException(
                status_code=500,
                detail="claude not found; set CLAUDE_BIN to the claude(.cmd) path or ensure it is on PATH",
            )
        # Claude CLI: claude -p "prompt" --model {model}
        # --system-prompt: 使用用户提供的 system prompt 或默认值
        # --tools "": 禁用所有工具，防止读取项目文件
        effective_system = system_prompt or "You are a helpful AI assistant. Respond directly to user requests."
        cmd = [CLAUDE_BIN, "-p", prompt, "--model", model, "--system-prompt", effective_system, "--tools", "", *args]
        backend_name = "claude"
    else:
        if not CODEX_BIN:
            raise HTTPException(
                status_code=500,
                detail="codex not found; set CODEX_BIN to the codex(.cmd) path or ensure it is on PATH",
            )
        # Codex CLI: codex exec -m {model} --skip-git-repo-check
        # 如果有 system_prompt，prepend 到 prompt 前面
        effective_prompt = f"System: {system_prompt} | {prompt}" if system_prompt else prompt
        cmd = [CODEX_BIN, "exec", "-m", model, *args, "--skip-git-repo-check"]
        prompt = effective_prompt  # 更新 prompt 用于 stdin
        backend_name = "codex"

    logger.info("Executing %s: cmd=%s timeout=%s", backend_name, cmd, timeout)

    async def _check_client_connected():
        """Check if client is still connected."""
        if request is not None:
            return not await request.is_disconnected()
        return True

    async def _acquire_slot():
        global _queued_waiters

        for attempt in range(MAX_RETRY_ATTEMPTS):
            # 检查客户端是否断开
            if not await _check_client_connected():
                logger.info("Client disconnected during queue wait, aborting")
                raise ClientDisconnectedError("Client disconnected")

            async with _queue_lock:
                if not MAX_QUEUE_SIZE or _queued_waiters < MAX_QUEUE_SIZE:
                    _queued_waiters += 1
                    queued = _queued_waiters
                    running = MAX_CONCURRENT_RUNS - getattr(EXEC_SEMAPHORE, "_value", 0)
                    logger.info("Enqueued request: queued=%s running=%s", queued, running)
                    break

            # 队列满，指数退避 + 抖动
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter
            logger.info("Queue full, retry %d/%d after %.2fs", attempt + 1, MAX_RETRY_ATTEMPTS, total_delay)

            # 分段 sleep，每 0.5 秒检查一次客户端连接
            sleep_interval = 0.5
            remaining = total_delay
            while remaining > 0:
                await asyncio.sleep(min(sleep_interval, remaining))
                remaining -= sleep_interval
                if not await _check_client_connected():
                    logger.info("Client disconnected during retry wait, aborting")
                    raise ClientDisconnectedError("Client disconnected")
        else:
            # 所有重试都失败
            logger.warning("Queue limit reached after %d retries", MAX_RETRY_ATTEMPTS)
            raise HTTPException(
                status_code=429,
                detail=f"Server busy: queue limit reached after {MAX_RETRY_ATTEMPTS} retries"
            )

        await EXEC_SEMAPHORE.acquire()
        async with _queue_lock:
            _queued_waiters -= 1
            running = MAX_CONCURRENT_RUNS - getattr(EXEC_SEMAPHORE, "_value", 0)
            logger.info("Acquired slot: queued=%s running=%s", _queued_waiters, running)

    await _acquire_slot()
    proc = None
    communicate_future = None
    try:
        # 准备环境变量
        env = os.environ.copy()
        user_home = os.path.expanduser("~")
        env.setdefault("HOME", user_home)
        env.setdefault("USERPROFILE", user_home)

        # 使用 subprocess.Popen（Windows 兼容）
        if use_claude:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            stdin_data = None
        else:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            stdin_data = prompt.encode("utf-8")

        # 在线程池中运行 communicate() 以避免管道死锁
        loop = asyncio.get_event_loop()
        communicate_future = loop.run_in_executor(
            None,  # 使用默认线程池
            lambda: proc.communicate(input=stdin_data)
        )

        # 等待完成，同时检查客户端连接和超时
        start_time = time.time()
        check_interval = 0.5

        while not communicate_future.done():
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("%s exec timed out: cmd=%s", backend_name, cmd)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                communicate_future.cancel()
                raise HTTPException(status_code=504, detail=f"{backend_name} exec timed out")

            # 检查客户端连接
            if not await _check_client_connected():
                logger.info("Client disconnected during execution, terminating subprocess")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                communicate_future.cancel()
                raise ClientDisconnectedError("Client disconnected")

            # 短暂等待
            await asyncio.sleep(check_interval)

        # 获取结果
        stdout, stderr = await communicate_future
        logger.info("%s subprocess completed", backend_name)

        raw_stdout = stdout.decode("utf-8", errors="replace")
        raw_stderr = stderr.decode("utf-8", errors="replace")

    except ClientDisconnectedError:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Command not found: {cmd[0]}")
    except subprocess.TimeoutExpired:
        if proc and proc.poll() is None:
            proc.kill()
        logger.warning("%s exec timed out: cmd=%s", backend_name, cmd)
        raise HTTPException(status_code=504, detail=f"{backend_name} exec timed out")
    except Exception as e:
        if proc and proc.poll() is None:
            proc.terminate()
        import traceback
        logger.error("%s unexpected error: %s\n%s", backend_name, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{backend_name} error: {str(e)}")
    finally:
        EXEC_SEMAPHORE.release()
        async with _queue_lock:
            running = MAX_CONCURRENT_RUNS - getattr(EXEC_SEMAPHORE, "_value", 0)
            logger.info("Released slot: queued=%s running=%s", _queued_waiters, running)

    logger.info(
        "%s result: returncode=%s stdout_len=%s stderr_len=%s",
        backend_name, proc.returncode, len(raw_stdout), len(raw_stderr)
    )
    if raw_stderr:
        logger.debug("%s stderr: %s", backend_name, raw_stderr[:500])
    output = raw_stdout.strip() if clean else raw_stdout
    return raw_stdout, raw_stderr, proc.returncode, output


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, request: Request):
    logger.info("/run model=%s args=%s", req.model, req.args)
    try:
        raw_stdout, raw_stderr, returncode, output = await _execute_llm(
            prompt=req.prompt,
            model=req.model,
            args=req.args,
            timeout=req.timeout,
            clean=req.clean,
            request=request,
        )
    except ClientDisconnectedError:
        logger.info("/run client disconnected, request cancelled")
        raise HTTPException(status_code=499, detail="Client disconnected")

    return RunResponse(
        output=output,
        stdout=None if req.clean else raw_stdout,
        stderr=None if req.clean else raw_stderr,
        returncode=returncode,
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest, request: Request, _: None = Depends(_require_token)):
    logger.info("/v1/completions model=%s", req.model)
    try:
        raw_stdout, raw_stderr, returncode, output = await _execute_llm(
            prompt=req.prompt,
            model=req.model,
            args=[],
            timeout=req.timeout,
            clean=True,
            request=request,
        )
    except ClientDisconnectedError:
        logger.info("/v1/completions client disconnected, request cancelled")
        raise HTTPException(status_code=499, detail="Client disconnected")
    # Claude CLI may return non-zero even on success; check output instead
    if returncode != 0 and not output:
        raise HTTPException(status_code=502, detail=raw_stderr.strip() or "LLM exec failed")

    now = int(time.time())
    prompt_tokens = _token_count(req.prompt)
    completion_tokens = _token_count(output)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        object="text_completion",
        created=now,
        model=req.model,
        choices=[
            CompletionChoice(
                index=0,
                text=output,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, request: Request, _: None = Depends(_require_token)):
    logger.info("/v1/chat/completions model=%s messages=%s", req.model, len(req.messages))
    prompt, system_prompt = _build_chat_prompt(req.messages)
    try:
        raw_stdout, raw_stderr, returncode, output = await _execute_llm(
            prompt=prompt,
            model=req.model,
            args=[],
            timeout=req.timeout,
            clean=True,
            request=request,
            system_prompt=system_prompt,
        )
    except ClientDisconnectedError:
        logger.info("/v1/chat/completions client disconnected, request cancelled")
        raise HTTPException(status_code=499, detail="Client disconnected")
    # Claude CLI may return non-zero even on success; check output instead
    if returncode != 0 and not output:
        raise HTTPException(status_code=502, detail=raw_stderr.strip() or "LLM exec failed")

    now = int(time.time())
    # token 计数包含 system_prompt
    total_prompt = f"{system_prompt} {prompt}" if system_prompt else prompt
    prompt_tokens = _token_count(total_prompt)
    completion_tokens = _token_count(output)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=now,
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChoiceMessage(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
