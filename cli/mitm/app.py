# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `m mitm` proxy server.

Fronts an OpenAI-compatible server. Chat-completion requests are shown to a hook,
which may answer them itself; everything else -- and every request the hook declines
-- is forwarded so that the client cannot tell the proxy is in the path.
"""

import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

try:
    import httpx
    import typer
    import uvicorn
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask
except ImportError as e:
    raise ImportError(
        "The 'm mitm' command requires extra dependencies. "
        'Please install them with: pip install "mellea[server]"'
    ) from e

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from mellea.core.utils import MelleaLogger
from mellea.helpers.openai_compatible_helpers import chat_completion_delta_merge

from .hooks import Hook, Reply, new_completion_id

logger = MelleaLogger.get_logger()

CHAT_COMPLETION_PATHS = ("/v1/chat/completions", "/chat/completions")
"""Request paths the hook is consulted for. Everything else is forwarded."""

PASSTHROUGH_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "content-length",
        "keep-alive",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
"""Headers that describe a single hop and must not be relayed to the next one.

Relaying a stale `content-length` is the classic way a rewriting proxy corrupts a
response body, and `transfer-encoding` is re-decided by the server framing our reply.
"""


def _forward_headers(headers: Any, upstream_host: str | None) -> dict[str, str]:
    """Build the header set to send upstream.

    Drops hop-by-hop headers and rewrites `host` to the upstream's, leaving
    `authorization` and any other credentials intact so upstream auth keeps working.

    Args:
        headers: Incoming request headers.
        upstream_host: Host header value for the upstream, or `None` to omit it.

    Returns:
        Headers safe to forward.
    """
    out = {
        k: v
        for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP
        and k.lower() != "host"
        and not k.lower().startswith("proxy-")
    }
    if upstream_host:
        out["host"] = upstream_host
    return out


def _response_headers(headers: Any) -> dict[str, str]:
    """Build the header set to send back to the client.

    Args:
        headers: Upstream response headers.

    Returns:
        Headers safe to relay, preserving `content-encoding` so that raw upstream
        bytes stay decodable by the client.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _upstream_url(upstream: str, path: str, query: str = "") -> str:
    """Map an incoming request path onto the upstream server.

    `upstream` is normally an origin such as `http://localhost:11434`, in which case
    the path is appended verbatim. A base path is also accepted: if `upstream` already
    ends with the start of the incoming path -- the common case of passing
    `http://localhost:11434/v1` -- that prefix is not duplicated.

    Args:
        upstream: Base URL of the server being fronted.
        path: Incoming request path, beginning with `/`.
        query: Incoming query string without the leading `?`. Defaults to empty.

    Returns:
        The absolute URL to send upstream.
    """
    base = upstream.rstrip("/")
    for depth in range(path.count("/"), 0, -1):
        prefix = "/" + "/".join(path.strip("/").split("/")[:depth])
        if base.endswith(prefix):
            base = base[: -len(prefix)]
            break
    url = f"{base}{path}"
    return f"{url}?{query}" if query else url


def resolve_hook(spec: str) -> Hook:
    """Import a hook from a `module:function` specification.

    Args:
        spec: Either `package.module:function` for an importable module, or
            `path/to/file.py:function` for a loose script.

    Returns:
        The resolved hook.

    Raises:
        ValueError: If `spec` has no `:` separator, names a file that cannot be
            loaded, or names an attribute that is missing or not callable.
        ImportError: If the module named by `spec` cannot be imported.
    """
    module_part, sep, attr = spec.rpartition(":")
    if not sep or not module_part or not attr:
        raise ValueError(
            f"Invalid hook {spec!r}. Expected 'module:function', "
            "for example 'cli.mitm.hooks:coin_flip'."
        )

    if module_part.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location("mitm_hook", module_part)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"Could not load hook module from file {module_part!r}.")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["mitm_hook"] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    hook = getattr(module, attr, None)
    if hook is None:
        raise ValueError(f"Hook module {module_part!r} has no attribute {attr!r}.")
    if not callable(hook):
        raise ValueError(f"Hook {spec!r} is not callable.")
    return cast(Hook, hook)


def _sse(chunk: ChatCompletionChunk) -> str:
    """Encode a chunk as a server-sent event.

    Args:
        chunk: The chunk to encode.

    Returns:
        An SSE `data:` frame, matching the wire format used by `cli/serve/streaming.py`.
    """
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


async def _stream_sse(chunks: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[str]:
    """Encode a chunk stream as SSE frames, terminated by the `[DONE]` sentinel.

    Args:
        chunks: The chunks to encode.

    Yields:
        SSE frames, then `data: [DONE]`.
    """
    async for chunk in chunks:
        yield _sse(chunk)
    yield "data: [DONE]\n\n"


async def _completion_to_chunks(
    completion: ChatCompletion,
) -> AsyncIterator[ChatCompletionChunk]:
    """Split a whole completion into the chunk sequence a streaming client expects.

    Args:
        completion: The completion to split.

    Yields:
        An opening chunk carrying the assistant role, a content chunk, and a final
        chunk carrying `finish_reason`.
    """
    choice = completion.choices[0]
    base: dict[str, Any] = {
        "id": completion.id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
    }

    def chunk(delta: dict[str, Any], finish: str | None) -> ChatCompletionChunk:
        return ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        )

    yield chunk({"role": "assistant", "content": ""}, None)

    delta: dict[str, Any] = {}
    if choice.message.content:
        delta["content"] = choice.message.content
    if choice.message.tool_calls:
        delta["tool_calls"] = [
            {"index": i, **tc.model_dump(exclude_unset=True)}
            for i, tc in enumerate(choice.message.tool_calls)
        ]
    if delta:
        yield chunk(delta, None)

    final = chunk({}, choice.finish_reason)
    if completion.usage is not None:
        final.usage = completion.usage
    yield final


async def _collapse_chunks(
    chunks: AsyncIterator[ChatCompletionChunk],
) -> ChatCompletion:
    """Assemble a chunk stream into the completion a non-streaming client expects.

    Delta merging -- including the index-keyed reassembly of tool-call fragments --
    is delegated to `chat_completion_delta_merge`. Its scratch fields are then dropped
    so the result is a clean OpenAI envelope rather than an internal representation.

    Args:
        chunks: The chunks to assemble.

    Returns:
        The assembled completion.

    Raises:
        ValueError: If the stream yielded no chunks.
    """
    deltas: list[dict[str, Any]] = []
    envelope: dict[str, Any] | None = None
    usage: Any = None

    async for chunk in chunks:
        raw = chunk.model_dump()
        if envelope is None:
            envelope = raw
        if raw.get("usage") is not None:
            usage = raw["usage"]
        deltas.extend(raw.get("choices") or [])

    if envelope is None:
        raise ValueError("Hook returned an empty chunk stream.")

    merged = chat_completion_delta_merge(deltas)
    message = merged["message"]

    # `chat_completion_delta_merge` always seeds these; only keep them if a delta
    # actually carried a value, so the reply matches what a real server sends.
    if not message.get("reasoning_content"):
        message.pop("reasoning_content", None)
    if not message.get("tool_calls"):
        message["tool_calls"] = None
    message.setdefault("role", "assistant")
    if message["role"] is None:
        message["role"] = "assistant"

    return ChatCompletion.model_validate(
        {
            "id": envelope.get("id") or new_completion_id(),
            "object": "chat.completion",
            "created": envelope["created"],
            "model": envelope["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": merged.get("finish_reason") or "stop",
                    "message": message,
                }
            ],
            **({"usage": usage} if usage is not None else {}),
        }
    )


async def _reply_response(reply: Reply, *, want_stream: bool) -> Response:
    """Render a hook's reply in the form the client asked for.

    A hook returns whichever form is natural to write; this reconciles that with the
    request's `stream` flag so hooks never have to branch on it.

    Args:
        reply: What the hook returned.
        want_stream: Whether the client requested a streamed response.

    Returns:
        An SSE `StreamingResponse` or a `JSONResponse`.
    """
    if want_stream:
        chunks = (
            _completion_to_chunks(reply) if isinstance(reply, ChatCompletion) else reply
        )
        return StreamingResponse(_stream_sse(chunks), media_type="text/event-stream")

    completion = (
        reply if isinstance(reply, ChatCompletion) else await _collapse_chunks(reply)
    )
    return JSONResponse(completion.model_dump(mode="json", exclude_unset=True))


def build_app(
    upstream: str,
    hook: Hook,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 600.0,
) -> FastAPI:
    """Build the proxy application.

    Args:
        upstream: Base URL of the OpenAI-compatible server to front.
        hook: Interceptor consulted for chat-completion requests.
        client: HTTP client used to reach the upstream. One is created and owned by
            the app if omitted; pass your own to redirect or stub the upstream, as
            the tests do.
        timeout: Read timeout in seconds for upstream requests, generous by default
            because generations are slow. Ignored when `client` is supplied.

    Returns:
        The configured application.
    """
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=False
    )
    upstream_host = httpx.URL(upstream).host or None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close the upstream client on shutdown, but only if this app created it."""
        yield
        if owns_client:
            await http.aclose()

    app = FastAPI(
        title="M mitm OpenAI API Compatible Proxy",
        description=f"Transparent intercepting proxy for {upstream}",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def forward(request: Request, body: bytes | None = None) -> Response:
        """Relay a request upstream and stream the reply back untouched."""
        payload = body if body is not None else await request.body()
        url = _upstream_url(upstream, request.url.path, request.url.query)
        upstream_request = http.build_request(
            request.method,
            url,
            headers=_forward_headers(request.headers, upstream_host),
            content=payload,
        )
        try:
            upstream_response = await http.send(upstream_request, stream=True)
        except httpx.HTTPError as e:
            logger.warning("Upstream request to %s failed: %s", url, e)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Upstream request failed: {e}",
                        "type": "upstream_error",
                    }
                },
            )
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=_response_headers(upstream_response.headers),
            background=BackgroundTask(upstream_response.aclose),
        )

    async def chat_completions(request: Request) -> Response:
        """Consult the hook, then either answer directly or forward upstream."""
        body = await request.body()

        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None

        if not isinstance(parsed, dict):
            # Not something a hook can reason about; let the upstream decide how to
            # reject it, so error behaviour matches too.
            return await forward(request, body)

        try:
            reply = hook(parsed)
            if inspect.isawaitable(reply):
                reply = await reply
        except Exception:
            logger.exception("Hook raised; forwarding request upstream instead")
            return await forward(request, body)

        if reply is None:
            return await forward(request, body)

        logger.info("Hook intercepted a request for model %s", parsed.get("model"))
        return await _reply_response(
            cast(Reply, reply), want_stream=bool(parsed.get("stream", False))
        )

    for path in CHAT_COMPLETION_PATHS:
        app.add_api_route(path, chat_completions, methods=["POST"])

    # Registered last so the chat-completion routes above win.
    app.add_api_route(
        "/{full_path:path}",
        cast(Callable[..., Any], forward),
        methods=PASSTHROUGH_METHODS,
    )

    return app


def run_server(
    upstream: str,
    hook: str = "cli.mitm.hooks:coin_flip",
    host: str = "0.0.0.0",
    port: int = 8081,
    timeout: float = 600.0,
) -> None:
    """Resolve the hook and run the proxy until interrupted.

    Args:
        upstream: Base URL of the OpenAI-compatible server to front.
        hook: Hook specification as `module:function`.
        host: Interface to bind.
        port: Port to bind.
        timeout: Read timeout in seconds for upstream requests.
    """
    resolved = resolve_hook(hook)
    app = build_app(upstream, resolved, timeout=timeout)
    typer.echo(f"Proxying http://{host}:{port} -> {upstream} (hook: {hook})")
    uvicorn.run(app, host=host, port=port)
