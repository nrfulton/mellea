# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `m mitm` proxy server.

Fronts an OpenAI-compatible server. Chat-completion requests are shown to a hook,
which may answer them itself; everything else -- and every request the hook declines
-- is forwarded so that the client cannot tell the proxy is in the path.

A *response* hook may also be registered, which sees the reply the upstream produced and
may replace it. That is the interception point for anything judged from the reply rather
than the request, such as the `reply_cannot_contain` restrictions of a `cli.mitm.policy`
policy. It has one cost: the upstream reply must be buffered in full before it can be
judged, so a streamed reply reaches the client all at once rather than token by token.
Without a response hook the byte-for-byte streaming passthrough is untouched.

Replies a *request* hook produced are not shown to the response hook. Those never came
from the upstream, so there is nothing about them to screen.
"""

import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
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
from pydantic import ValidationError

from mellea.core.utils import MelleaLogger
from mellea.helpers.openai_compatible_helpers import chat_completion_delta_merge

from .hooks import Hook, Reply, ResponseHook, new_completion_id
from .policy import PolicyRegistry

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


def _forward_headers(
    headers: Any, upstream_host: str | None, *, identity_encoding: bool = False
) -> dict[str, str]:
    """Build the header set to send upstream.

    Drops hop-by-hop headers and rewrites `host` to the upstream's, leaving
    `authorization` and any other credentials intact so upstream auth keeps working.

    Args:
        headers: Incoming request headers.
        upstream_host: Host header value for the upstream, or `None` to omit it.
        identity_encoding: Ask the upstream not to compress its reply. Set when the
            reply will be parsed rather than relayed, so that the body arrives as bytes
            this process can read without decompressing them first.

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
    if identity_encoding:
        out.pop("Accept-Encoding", None)
        out.pop("accept-encoding", None)
        out["accept-encoding"] = "identity"
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


async def _parse_sse_chunks(body: bytes) -> AsyncIterator[ChatCompletionChunk]:
    """Read chunks back out of a buffered SSE response body.

    The inverse of `_sse`, tolerant in the way a proxy has to be: `CRLF` framing is
    accepted as well as `LF`, frames are split on blank lines, anything that is not a
    `data:` line is ignored, the `[DONE]` sentinel is skipped, and a frame that does not
    parse as a chunk is dropped rather than failing the whole reply.

    Args:
        body: The complete response body from an upstream that streamed.

    Yields:
        Each chunk the body carried, in wire order.
    """
    text = body.decode("utf-8", errors="replace").replace("\r\n", "\n")
    for frame in text.split("\n\n"):
        payload = "".join(
            line[len("data:") :].strip()
            for line in frame.splitlines()
            if line.startswith("data:")
        )
        if not payload or payload == "[DONE]":
            continue
        try:
            yield ChatCompletionChunk.model_validate_json(payload)
        except (ValidationError, ValueError):
            logger.debug("Skipping unparseable SSE frame from upstream")


async def _completion_from_body(
    body: bytes, content_type: str
) -> ChatCompletion | None:
    """Assemble an upstream reply body into a completion a response hook can read.

    Args:
        body: The complete upstream response body.
        content_type: The upstream's `content-type`, used to tell a streamed reply from
            a whole one. A body starting with `data:` is treated as streamed even if the
            header says otherwise.

    Returns:
        The assembled completion, or `None` if the body is not a chat completion this
        proxy can represent -- in which case the caller must relay it untouched rather
        than screen it.
    """
    if "text/event-stream" in content_type.lower() or body.lstrip().startswith(
        b"data:"
    ):
        try:
            return await _collapse_chunks(_parse_sse_chunks(body))
        except (ValidationError, ValueError):
            logger.debug("Upstream SSE body did not assemble into a completion")
            return None

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ChatCompletion.model_validate(payload)
    except ValidationError:
        logger.debug("Upstream JSON body is not a chat completion")
        return None


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


def _bad_gateway(url: str, error: Exception) -> JSONResponse:
    """Report an upstream that could not be reached.

    Args:
        url: The upstream URL that failed.
        error: The transport error raised.

    Returns:
        A 502 in the OpenAI error envelope.
    """
    logger.warning("Upstream request to %s failed: %s", url, error)
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": f"Upstream request failed: {error}",
                "type": "upstream_error",
            }
        },
    )


def build_app(
    upstream: str,
    hook: Hook,
    *,
    response_hook: ResponseHook | None = None,
    policies: PolicyRegistry | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 600.0,
) -> FastAPI:
    """Build the proxy application.

    Args:
        upstream: Base URL of the OpenAI-compatible server to front.
        hook: Interceptor consulted for chat-completion requests.
        response_hook: Interceptor consulted for the replies the upstream produces, or
            `None` to relay them untouched. Registering one makes the proxy buffer each
            upstream reply so that it can be judged whole, which costs a streamed reply
            its incrementality.
        policies: Policy registry the app exposes as `app.state.policies` and injects
            into a response hook that declares a `policies` parameter. A fresh empty
            registry is created if omitted; add to it at any time and the next request is
            screened against it.
        client: HTTP client used to reach the upstream. One is created and owned by
            the app if omitted; pass your own to redirect or stub the upstream, as
            the tests do.
        timeout: Read timeout in seconds for upstream requests, generous by default
            because generations are slow. Ignored when `client` is supplied.

    Returns:
        The configured application, with the policy registry on `app.state.policies`.
    """
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=False
    )
    upstream_host = httpx.URL(upstream).host or None
    registry = policies if policies is not None else PolicyRegistry()

    # Decided once at build time rather than per request: a response hook that wants the
    # registry says so by declaring a `policies` parameter.
    wants_policies = False
    if response_hook is not None:
        try:
            wants_policies = "policies" in inspect.signature(response_hook).parameters
        except (TypeError, ValueError):  # builtins and C callables have no signature
            wants_policies = False

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
    app.state.policies = registry

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
            return _bad_gateway(url, e)
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=_response_headers(upstream_response.headers),
            background=BackgroundTask(upstream_response.aclose),
        )

    async def screen(request: Request, body: bytes, parsed: dict[str, Any]) -> Response:
        """Fetch the upstream reply in full, show it to the response hook, and answer.

        Only reached when a response hook is registered. Everything the hook cannot or
        does not want to act on is relayed with the upstream's own status and headers, so
        error behaviour and vendor extension fields survive the round trip.
        """
        url = _upstream_url(upstream, request.url.path, request.url.query)
        upstream_request = http.build_request(
            request.method,
            url,
            headers=_forward_headers(
                request.headers, upstream_host, identity_encoding=True
            ),
            content=body,
        )
        try:
            upstream_response = await http.send(upstream_request)
        except httpx.HTTPError as e:
            return _bad_gateway(url, e)

        raw = upstream_response.content
        relay = Response(
            content=raw,
            status_code=upstream_response.status_code,
            headers=_response_headers(upstream_response.headers),
        )

        if upstream_response.status_code >= 300:
            return relay

        completion = await _completion_from_body(
            raw, upstream_response.headers.get("content-type", "")
        )
        if completion is None:
            return relay

        assert response_hook is not None  # only called when one is registered
        try:
            screened = (
                response_hook(parsed, completion, policies=registry)
                if wants_policies
                else response_hook(parsed, completion)
            )
            if inspect.isawaitable(screened):
                screened = await screened
        except Exception:
            logger.exception(
                "Response hook raised; relaying the upstream reply instead"
            )
            return relay

        if screened is None:
            return relay

        logger.info("Response hook replaced a reply for model %s", completion.model)
        return await _reply_response(
            cast(Reply, screened), want_stream=bool(parsed.get("stream", False))
        )

    async def chat_completions(request: Request) -> Response:
        """Consult the hooks, then either answer directly, screen, or forward upstream."""
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
            reply = None

        if reply is not None:
            logger.info("Hook intercepted a request for model %s", parsed.get("model"))
            return await _reply_response(
                cast(Reply, reply), want_stream=bool(parsed.get("stream", False))
            )

        if response_hook is None:
            return await forward(request, body)
        return await screen(request, body, parsed)

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
    response_hook: str | None = None,
    policies: Sequence[str | Path] = (),
) -> None:
    """Resolve the hooks, load any policies, and run the proxy until interrupted.

    Policies are loaded before the server binds, so the proxy is enforcing from its first
    request rather than from whenever the first policy happens to arrive.

    Args:
        upstream: Base URL of the OpenAI-compatible server to front.
        hook: Hook specification as `module:function`.
        host: Interface to bind.
        port: Port to bind.
        timeout: Read timeout in seconds for upstream requests.
        response_hook: Response-hook specification as `module:function`, or `None` to
            relay upstream replies untouched.
        policies: Policy files, or directories of them, to load into the registry.

    Raises:
        PolicyError: If a policy path does not exist or does not parse. Raised before the
            server binds, so a bad policy fails the command rather than a request.
    """
    registry = PolicyRegistry()
    for path in policies:
        for policy in registry.add_path(path):
            typer.echo(f"Loaded policy {policy.risk_group} ({len(policy.risks)} risks)")

    resolved_response_hook = (
        resolve_hook(response_hook) if response_hook is not None else None
    )
    app = build_app(
        upstream,
        resolve_hook(hook),
        response_hook=resolved_response_hook,
        policies=registry,
        timeout=timeout,
    )
    hooks = hook if response_hook is None else f"{hook}, response: {response_hook}"
    typer.echo(f"Proxying http://{host}:{port} -> {upstream} (hook: {hooks})")
    uvicorn.run(app, host=host, port=port)
