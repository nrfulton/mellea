# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy server implementation that forwards OpenAI chat requests to an upstream endpoint.

This module provides the FastAPI server that proxies requests to another OpenAI-compatible
endpoint, optionally applying request/response transformations via an MProxy implementation.
"""

import importlib
from collections.abc import AsyncIterator

import httpx
import typer
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cli.proxy.mproxy import MProxy, PassthroughProxy

app = FastAPI(
    title="Mellea Proxy Server",
    description="OpenAI-compatible proxy with request/response rewriting",
    version="0.1.0",
)

# Module-level state set by run_proxy_server
_upstream_endpoint: str = ""
_mproxy: MProxy = PassthroughProxy()
_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the HTTP client on server startup."""
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close the HTTP client on server shutdown."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic liveness check endpoint."""
    return {"status": "pass"}


async def _stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    """Stream chunks from upstream, applying chunk_rewrite transformations."""
    from openai.types.chat import ChatCompletionChunk

    async for line in response.aiter_lines():
        if not line:
            yield b"\n"
            continue

        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                yield b"data: [DONE]\n\n"
                continue

            try:
                import json

                chunk_dict = json.loads(data)
                chunk = ChatCompletionChunk.model_validate(chunk_dict)
                rewritten = _mproxy.chunk_rewrite(chunk)
                yield f"data: {rewritten.model_dump_json()}\n\n".encode()
            except Exception:
                # If parsing fails, pass through unchanged
                yield f"{line}\n".encode()
        else:
            yield f"{line}\n".encode()


@app.api_route("/v1/chat/completions", methods=["POST"], include_in_schema=True)
async def proxy_chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Proxy chat completion requests to the upstream endpoint.

    Applies request_rewrite before sending and response_rewrite/chunk_rewrite
    to the response.
    """
    from openai.types.chat import ChatCompletion

    if _http_client is None:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "HTTP client not initialized"}},
        )

    # Parse the incoming request body
    body = await request.json()

    # Apply request rewrite
    rewritten_request = _mproxy.request_rewrite(body)

    # Determine if streaming
    is_streaming = rewritten_request.get("stream", False)

    # Forward headers (excluding host and content-length which httpx handles)
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    upstream_url = f"{_upstream_endpoint}/chat/completions"

    if is_streaming:
        # For streaming, we need to stream the response back
        upstream_response = await _http_client.post(
            upstream_url, json=rewritten_request, headers=forward_headers
        )

        if upstream_response.status_code != 200:
            content = await upstream_response.aread()
            return JSONResponse(
                status_code=upstream_response.status_code,
                content=content.decode() if content else {"error": "Upstream error"},
            )

        return StreamingResponse(
            _stream_response(upstream_response),
            media_type="text/event-stream",
            headers={
                k: v
                for k, v in upstream_response.headers.items()
                if k.lower()
                not in ("content-length", "transfer-encoding", "content-encoding")
            },
        )
    else:
        # Non-streaming request
        upstream_response = await _http_client.post(
            upstream_url, json=rewritten_request, headers=forward_headers
        )

        if upstream_response.status_code != 200:
            return JSONResponse(
                status_code=upstream_response.status_code,
                content=upstream_response.json(),
            )

        # Parse and apply response rewrite
        response_data = upstream_response.json()
        completion = ChatCompletion.model_validate(response_data)
        rewritten = _mproxy.response_rewrite(completion)

        return JSONResponse(status_code=200, content=rewritten.model_dump(mode="json"))


def load_mproxy(mproxy_path: str) -> MProxy:
    """Load an MProxy implementation from a dotted path.

    Args:
        mproxy_path: Dotted path to the MProxy class (e.g., 'mymodule.MyProxy').

    Returns:
        An instance of the MProxy implementation.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class doesn't exist in the module.
        TypeError: If the class is not a subclass of MProxy.
    """
    module_path, class_name = mproxy_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type) or not issubclass(cls, MProxy):
        raise TypeError(f"{mproxy_path} is not a subclass of MProxy")

    return cls()


def run_proxy_server(
    endpoint: str, host: str = "0.0.0.0", port: int = 8080, mproxy: MProxy | None = None
) -> None:
    """Start the proxy server.

    Args:
        endpoint: The upstream OpenAI-compatible endpoint URL (e.g., 'http://127.0.0.1:7999/v1').
        host: Host to bind to.
        port: Port to bind to.
        mproxy: Optional MProxy implementation for request/response rewriting.
    """
    global _upstream_endpoint, _mproxy

    _upstream_endpoint = endpoint.rstrip("/")
    _mproxy = mproxy or PassthroughProxy()

    typer.echo(
        f"Proxying /v1/chat/completions to {_upstream_endpoint}/chat/completions"
    )
    typer.echo(f"MProxy: {_mproxy.__class__.__name__}")
    typer.echo(f"Listening at http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)
