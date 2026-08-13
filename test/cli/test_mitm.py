# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `m mitm` intercepting proxy.

The upstream server is a stub FastAPI app reached over `httpx.ASGITransport`, so these
tests exercise the real proxy code path -- header handling, streaming, passthrough --
without opening a socket.
"""

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from cli.mitm.app import build_app, resolve_hook
from cli.mitm.hooks import REFUSAL, chunks_from_text, coin_flip, completion_from_text

UPSTREAM_TEXT = "The pod bay doors are open."

# A field no OpenAI-compatible schema knows about. Real providers add these (vLLM's
# `reasoning_content`, for instance), and a transparent proxy must not eat them.
VENDOR_FIELD = "reasoning_content"


def upstream_completion() -> dict[str, Any]:
    """Build the completion body the stub upstream returns."""
    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": UPSTREAM_TEXT,
                    VENDOR_FIELD: "thinking out loud",
                },
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
        "upstream_only_field": {"nested": True},
    }


@pytest.fixture
def upstream_calls() -> list[dict[str, Any]]:
    """Record of every request the stub upstream received."""
    return []


@pytest.fixture
def upstream_app(upstream_calls: list[dict[str, Any]]) -> FastAPI:
    """A minimal stand-in for a real OpenAI-compatible server."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            # A real server rejects this; recording it proves the proxy forwarded
            # the unparsable body rather than swallowing it.
            upstream_calls.append({"path": "/v1/chat/completions", "body": raw})
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "invalid JSON",
                        "type": "invalid_request_error",
                    }
                },
            )
        upstream_calls.append({"path": "/v1/chat/completions", "body": body})

        if body.get("stream"):

            async def events():
                for piece in ("The pod ", "bay doors ", "are open."):
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": "chatcmpl-upstream",
                                "object": "chat.completion.chunk",
                                "created": 1700000000,
                                "model": "upstream-model",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": piece},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return JSONResponse(upstream_completion())

    @app.get("/v1/models")
    async def models() -> Any:
        upstream_calls.append({"path": "/v1/models", "body": None})
        return JSONResponse(
            {"object": "list", "data": [{"id": "upstream-model", "object": "model"}]}
        )

    return app


@pytest.fixture
def make_proxy(upstream_app: FastAPI):
    """Build a `TestClient` for the proxy, wired to the stub upstream."""

    def _make(hook) -> TestClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app), base_url="http://upstream"
        )
        return TestClient(build_app("http://upstream", hook, client=client))

    return _make


def sse_payloads(text: str) -> list[dict[str, Any]]:
    """Parse an SSE response body into its JSON frames, excluding `[DONE]`."""
    frames = [
        line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]", f"stream did not terminate with [DONE]: {frames}"
    return [json.loads(f) for f in frames[:-1]]


def test_passthrough_is_byte_identical(make_proxy, upstream_calls):
    """A declining hook leaves a non-streamed response untouched, vendor fields included."""
    with make_proxy(lambda request: None) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json() == upstream_completion()
    # The field the repo's own strict ChatCompletion model would have dropped.
    assert response.json()["choices"][0]["message"][VENDOR_FIELD] == "thinking out loud"
    assert response.json()["upstream_only_field"] == {"nested": True}
    assert len(upstream_calls) == 1


def test_passthrough_forwards_request_body_unchanged(make_proxy, upstream_calls):
    """The upstream sees exactly what the client sent, extra keys and all."""
    sent = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.25,
        "chat_template_kwargs": {"thinking": True},
    }
    with make_proxy(lambda request: None) as client:
        client.post("/v1/chat/completions", json=sent)

    assert upstream_calls[0]["body"] == sent


def test_passthrough_streaming(make_proxy, upstream_calls):
    """A declining hook relays the upstream's SSE stream."""
    with make_proxy(lambda request: None) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    frames = sse_payloads(response.text)
    content = "".join(f["choices"][0]["delta"]["content"] for f in frames)
    assert content == UPSTREAM_TEXT
    assert len(upstream_calls) == 1


def test_hook_chunks_to_streaming_client(make_proxy, upstream_calls):
    """An intercepted streaming request never reaches the upstream."""

    def hook(request):
        return chunks_from_text(REFUSAL, model=request["model"])

    with make_proxy(hook) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hal-9000",
                "messages": [{"role": "user", "content": "open the doors"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = sse_payloads(response.text)
    content = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
    assert content == REFUSAL
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(f["model"] == "hal-9000" for f in frames)
    # More than one content chunk, i.e. the stream is genuinely chunked.
    assert sum(1 for f in frames if f["choices"][0]["delta"].get("content")) > 1
    assert upstream_calls == []


def test_hook_chunks_to_nonstreaming_client(make_proxy, upstream_calls):
    """Chunks are collapsed into a single completion when the client did not ask to stream."""

    def hook(request):
        return chunks_from_text(REFUSAL, model=request["model"])

    with make_proxy(hook) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "hal-9000", "messages": [{"role": "user", "content": "x"}]},
        )

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == REFUSAL
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["model"] == "hal-9000"
    # Internal scratch fields from the delta merge must not leak onto the wire.
    assert "parsed" not in body["choices"][0]["message"]
    assert "reasoning_content" not in body["choices"][0]["message"]
    assert upstream_calls == []


def test_hook_completion_to_nonstreaming_client(make_proxy, upstream_calls):
    """A whole completion from the hook is sent as JSON."""
    with make_proxy(
        lambda request: completion_from_text(REFUSAL, model=request["model"])
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "hal-9000", "messages": [{"role": "user", "content": "x"}]},
        )

    body = response.json()
    assert body["choices"][0]["message"]["content"] == REFUSAL
    assert body["object"] == "chat.completion"
    assert upstream_calls == []


def test_hook_completion_to_streaming_client(make_proxy, upstream_calls):
    """A whole completion from the hook is chunked for a streaming client."""
    with make_proxy(
        lambda request: completion_from_text(REFUSAL, model=request["model"])
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hal-9000",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )

    frames = sse_payloads(response.text)
    content = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
    assert content == REFUSAL
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert upstream_calls == []


async def test_async_hook_is_awaited(make_proxy, upstream_calls):
    """A coroutine hook works the same as a plain one."""

    async def hook(request):
        return completion_from_text(REFUSAL, model=request["model"])

    with make_proxy(hook) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )

    assert response.json()["choices"][0]["message"]["content"] == REFUSAL
    assert upstream_calls == []


def test_non_chat_path_passes_through(make_proxy, upstream_calls):
    """Paths the hook never sees behave exactly like the upstream."""
    with make_proxy(lambda request: completion_from_text("nope", "m")) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "upstream-model", "object": "model"}],
    }
    assert upstream_calls == [{"path": "/v1/models", "body": None}]


def test_hook_exception_falls_back_to_upstream(make_proxy, upstream_calls):
    """A broken hook must not take the endpoint down with it."""

    def hook(request):
        raise RuntimeError("hook is broken")

    with make_proxy(hook) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )

    assert response.status_code == 200
    assert response.json() == upstream_completion()
    assert len(upstream_calls) == 1


def test_coin_flip_refuses_on_heads(monkeypatch):
    """The example hook returns a chunk stream when the coin comes up heads."""
    monkeypatch.setattr("cli.mitm.hooks.random.random", lambda: 0.0)
    assert coin_flip({"model": "m"}) is not None


def test_coin_flip_declines_on_tails(monkeypatch):
    """The example hook returns None when the coin comes up tails."""
    monkeypatch.setattr("cli.mitm.hooks.random.random", lambda: 0.99)
    assert coin_flip({"model": "m"}) is None


def test_coin_flip_end_to_end(make_proxy, upstream_calls, monkeypatch):
    """Both coin_flip branches produce a valid reply through the whole proxy."""
    monkeypatch.setattr("cli.mitm.hooks.random.random", lambda: 0.0)
    with make_proxy(coin_flip) as client:
        refused = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
    assert refused.json()["choices"][0]["message"]["content"] == REFUSAL
    assert upstream_calls == []

    monkeypatch.setattr("cli.mitm.hooks.random.random", lambda: 0.99)
    with make_proxy(coin_flip) as client:
        allowed = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
    assert allowed.json() == upstream_completion()
    assert len(upstream_calls) == 1


def test_malformed_body_is_forwarded(make_proxy, upstream_calls):
    """A body the hook cannot parse is left for the upstream to reject."""
    hook_calls = []

    def hook(request):
        hook_calls.append(request)
        return completion_from_text("nope", "m")

    with make_proxy(hook) as client:
        response = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"content-type": "application/json"},
        )

    assert hook_calls == []
    # Forwarded verbatim, and the upstream's rejection reached the client unaltered.
    assert upstream_calls == [{"path": "/v1/chat/completions", "body": b"not json"}]
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_resolve_hook_dotted_module():
    """A `module:function` spec resolves to the callable."""
    assert resolve_hook("cli.mitm.hooks:coin_flip") is coin_flip


def test_resolve_hook_from_file(tmp_path):
    """A `file.py:function` spec resolves to the callable."""
    script = tmp_path / "my_hook.py"
    script.write_text("def block(request):\n    return None\n")
    assert resolve_hook(f"{script}:block").__name__ == "block"


@pytest.mark.parametrize(
    "spec, match",
    [
        ("cli.mitm.hooks", "Expected 'module:function'"),
        ("cli.mitm.hooks:missing_attr", "has no attribute"),
        ("cli.mitm.hooks:REFUSAL", "not callable"),
    ],
)
def test_resolve_hook_rejects_bad_specs(spec, match):
    """Bad hook specifications fail with an actionable message."""
    with pytest.raises(ValueError, match=match):
        resolve_hook(spec)
