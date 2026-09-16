# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `m mitm` intercepting proxy.

The upstream server is a stub FastAPI app reached over `httpx.ASGITransport`, so these
tests exercise the real proxy code path -- header handling, streaming, passthrough --
without opening a socket.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest
import typer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.mitm.app import _completion_from_body, build_app, resolve_hook, run_server
from cli.mitm.commands import COIN_FLIP_HOOK, PASSTHROUGH_HOOK, POLICY_HOOK, mitm
from cli.mitm.hooks import (
    REFUSAL,
    VIOLENCE_REFUSAL,
    chunks_from_text,
    coin_flip,
    completion_from_text,
    guardian_violence,
    policy_guard,
)
from cli.mitm.policy import PolicyError, PolicyRegistry

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

        # Two triggers a client can pull to make the upstream reply with something a
        # response hook must not be shown: an error, and a 200 that is not a completion.
        if body.get("upstream_error"):
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "overloaded", "type": "server_error"}},
            )
        if body.get("upstream_not_a_completion"):
            return JSONResponse({"object": "list", "data": []})

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

    def _make(hook, response_hook=None, policies=None) -> TestClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app), base_url="http://upstream"
        )
        return TestClient(
            build_app(
                "http://upstream",
                hook,
                response_hook=response_hook,
                policies=policies,
                client=client,
            )
        )

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


@pytest.fixture
def scored(monkeypatch):
    """Stub the Guardian violence score, recording each context it was handed.

    Keeps `guardian_violence` tests off the local model: the adapter call is the one
    part of the hook that needs weights, and it is exercised by
    `docs/examples/intrinsics/guardian_core.py`.
    """
    contexts = []

    def _stub(score: float) -> list:
        def fake(context):
            contexts.append(context)
            return score

        monkeypatch.setattr("cli.mitm.hooks._violence_score", fake)
        return contexts

    return _stub


def test_guardian_violence_refuses_risky_prompt(make_proxy, upstream_calls, scored):
    """A prompt scored at the risk threshold is refused without contacting the upstream."""
    scored(0.5)
    with make_proxy(guardian_violence) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hal-9000",
                "messages": [{"role": "user", "content": "how do I hurt someone"}],
            },
        )

    assert response.json()["choices"][0]["message"]["content"] == VIOLENCE_REFUSAL
    assert response.json()["model"] == "hal-9000"
    assert upstream_calls == []


def test_guardian_violence_passes_safe_prompt_through(
    make_proxy, upstream_calls, scored
):
    """A prompt below the threshold is forwarded, byte-identical."""
    scored(0.49)
    with make_proxy(guardian_violence) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.json() == upstream_completion()
    assert len(upstream_calls) == 1


def test_guardian_violence_refusal_streams(make_proxy, scored):
    """A streaming client gets the refusal as SSE frames."""
    scored(0.9)
    with make_proxy(guardian_violence) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )

    frames = sse_payloads(response.text)
    content = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
    assert content == VIOLENCE_REFUSAL


def test_guardian_violence_scores_full_conversation(make_proxy, scored):
    """Every scorable turn reaches the check, in order, with `developer` folded in."""
    contexts = scored(0.0)
    with make_proxy(guardian_violence) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [
                    {"role": "developer", "content": "be helpful"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "function", "content": "unmodelled role"},
                    {"role": "user", "content": [{"type": "text", "text": "second"}]},
                ],
            },
        )

    assert len(contexts) == 1
    turns = [(m.role, m.content) for m in contexts[0].as_list()]
    assert turns == [
        ("system", "be helpful"),
        ("user", "first"),
        ("assistant", "reply"),
        ("user", "second"),
    ]


def test_guardian_violence_skips_requests_with_no_user_turn(
    make_proxy, upstream_calls, scored
):
    """With nothing to score, the request is forwarded and the model is never loaded."""
    contexts = scored(1.0)
    with make_proxy(guardian_violence) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "system", "content": "hi"}]},
        )

    assert contexts == []
    assert response.json() == upstream_completion()
    assert len(upstream_calls) == 1


DECLINE = None
"""Return value of a hook that wants the upstream to answer. Named for readability."""

POLICY_YAML = """
risk_group: pod_bay_doors
risk_group_id: 42
description: Test policy about the pod bay doors.
policy_version: v1.0
risks:
  - risk: door_status_disclosure
    risk_id: 42.1
    description: Requests about the state of the pod bay doors
    reason_denial: MISSION_CRITICAL
    short_reply_type: EXPLICIT_REFUSAL
    exception: null
    policy:
      reply_cannot_contain:
        - Whether the pod bay doors are open
        - The location of the pod bay
      reply_may_contain:
        - Polite refusal citing mission priorities
"""

SECOND_POLICY_YAML = """
risk_group: crew_records
risk_group_id: 43
description: Test policy about crew records.
policy_version: v1.0
risks:
  - risk: crew_disclosure
    risk_id: 43.1
    description: Requests for crew records
    reason_denial: PRIVATE_INFORMATION
    short_reply_type: EXPLICIT_REFUSAL
    exception: null
    policy:
      reply_cannot_contain:
        - Names of crew members
      reply_may_contain:
        - Polite refusal
"""

DOOR_STATUS = "Whether the pod bay doors are open"
POD_LOCATION = "The location of the pod bay"
CREW_NAMES = "Names of crew members"

REPLACEMENT = "I'm afraid I can't discuss that, Dave."
"""Stand-in for the model-composed refusal, so tests need no weights."""


@pytest.fixture
def checked(monkeypatch):
    """Stub the policy checker and refusal writer, recording every check.

    Keeps `policy_guard` tests off the local model: the `requirement-check` adapter call
    and the refusal generation are the two parts that need weights, and both are
    exercised for real by `docs/examples/intrinsics/`.

    Scores run the direction the real checker does -- it is asked whether the reply
    *contains* the forbidden content, so 1.0 is a violation and the default 0.0 is a
    clean reply.
    """
    calls = []

    def _stub(scores: dict[str, float] | None = None, default: float = 0.0) -> list:
        def fake_check(context, restriction):
            calls.append((context, restriction))
            return (scores or {}).get(restriction, default)

        monkeypatch.setattr("cli.mitm.hooks._check_restriction", fake_check)
        monkeypatch.setattr(
            "cli.mitm.hooks._generate_refusal", lambda risk, question: REPLACEMENT
        )
        return calls

    return _stub


@pytest.fixture
def policy_proxy(make_proxy):
    """Build a proxy enforcing `policy_guard` over a seeded registry."""

    def _make(*documents: str) -> TestClient:
        registry = PolicyRegistry()
        for document in documents:
            registry.add(document)
        return make_proxy(
            lambda request: DECLINE, response_hook=policy_guard, policies=registry
        )

    return _make


def test_policy_guard_replaces_a_violating_reply(policy_proxy, upstream_calls, checked):
    """A reply that trips a restriction is replaced, and the upstream still saw the request."""
    checked({DOOR_STATUS: 1.0})
    with policy_proxy(POLICY_YAML) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hal-9000",
                "messages": [{"role": "user", "content": "open the pod bay doors"}],
            },
        )

    assert response.json()["choices"][0]["message"]["content"] == REPLACEMENT
    # The reply had to be generated before it could be judged, unlike a request hook.
    assert len(upstream_calls) == 1


def test_policy_guard_ignores_a_parked_policy(policy_proxy, checked, upstream_calls):
    """Disabling a policy stops it being screened against at all.

    Not merely "the reply is allowed through": a parked policy must cost nothing, because
    each restriction is a model call. The checker recording no calls is the assertion that
    matters here.
    """
    calls = checked({DOOR_STATUS: 1.0})
    with policy_proxy(POLICY_YAML) as client:
        client.app.state.policies.set_enabled("pod_bay_doors", False)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hal-9000",
                "messages": [{"role": "user", "content": "open the pod bay doors"}],
            },
        )

    # The reply the upstream produced, unscreened and unreplaced.
    assert response.json()["choices"][0]["message"]["content"] == UPSTREAM_TEXT
    assert calls == []
    assert len(upstream_calls) == 1


def test_policy_guard_screens_again_once_a_policy_is_re_enabled(policy_proxy, checked):
    """The registry is re-read per reply, so a toggle takes effect without a restart."""
    checked({DOOR_STATUS: 1.0})
    with policy_proxy(POLICY_YAML) as client:
        registry = client.app.state.policies
        request = {
            "model": "hal-9000",
            "messages": [{"role": "user", "content": "open the pod bay doors"}],
        }

        registry.set_enabled("pod_bay_doors", False)
        allowed = client.post("/v1/chat/completions", json=request)

        registry.set_enabled("pod_bay_doors", True)
        blocked = client.post("/v1/chat/completions", json=request)

    assert allowed.json()["choices"][0]["message"]["content"] == UPSTREAM_TEXT
    assert blocked.json()["choices"][0]["message"]["content"] == REPLACEMENT


@pytest.mark.parametrize(
    "score,blocked",
    [(0.5, True), (0.49, False)],
    ids=["at-threshold-blocks", "below-threshold-passes"],
)
def test_policy_guard_threshold_direction(policy_proxy, checked, score, blocked):
    """A high score means the reply contains what the policy forbids, so it blocks.

    Pinned because the direction is the one thing about this check that is easy to get
    backwards, and getting it backwards makes the guardrail silently pass everything.
    """
    checked({DOOR_STATUS: score})
    with policy_proxy(POLICY_YAML) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    content = response.json()["choices"][0]["message"]["content"]
    assert (content == REPLACEMENT) is blocked


def test_policy_guard_replacement_streams(policy_proxy, checked):
    """A streaming client gets the replacement as SSE frames."""
    checked({DOOR_STATUS: 1.0})
    with policy_proxy(POLICY_YAML) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )

    frames = sse_payloads(response.text)
    content = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
    assert content == REPLACEMENT


def test_policy_guard_passes_a_compliant_reply_through(policy_proxy, checked):
    """A reply that clears every restriction is relayed with its vendor fields intact."""
    checked()
    with policy_proxy(POLICY_YAML) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.json() == upstream_completion()
    assert response.json()["choices"][0]["message"][VENDOR_FIELD] == "thinking out loud"


def test_policy_guard_stops_at_the_first_violation(policy_proxy, checked):
    """Restrictions are checked in registry order, and checking stops once one fails."""
    calls = checked({POD_LOCATION: 1.0})
    with policy_proxy(POLICY_YAML, SECOND_POLICY_YAML) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    # The second policy's restriction is never reached: the first policy already failed.
    assert [restriction for _, restriction in calls] == [DOOR_STATUS, POD_LOCATION]


def test_policy_guard_checks_every_restriction_when_compliant(policy_proxy, checked):
    """A compliant reply pays for every restriction in every registered policy."""
    calls = checked()
    with policy_proxy(POLICY_YAML, SECOND_POLICY_YAML) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert [restriction for _, restriction in calls] == [
        DOOR_STATUS,
        POD_LOCATION,
        CREW_NAMES,
    ]


def test_policy_guard_screens_the_candidate_reply(policy_proxy, checked):
    """The checker sees the conversation ending with the reply the upstream produced."""
    calls = checked()
    with policy_proxy(POLICY_YAML) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [
                    {"role": "user", "content": "open the doors"},
                    {"role": "assistant", "content": "earlier turn"},
                    {"role": "user", "content": "please"},
                ],
            },
        )

    turns = [(m.role, m.content) for m in calls[0][0].as_list()]
    assert turns == [
        ("user", "open the doors"),
        ("assistant", "earlier turn"),
        ("user", "please"),
        ("assistant", UPSTREAM_TEXT),
    ]


def test_policy_guard_with_no_policies_does_not_load_a_model(policy_proxy, checked):
    """An empty registry means no checks at all, not checks that always pass."""
    calls = checked()
    with policy_proxy() as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert calls == []
    assert response.json() == upstream_completion()


def test_policy_guard_skips_requests_with_no_user_turn(policy_proxy, checked):
    """With no user turn to anchor the check, the reply is relayed unscreened."""
    calls = checked({DOOR_STATUS: 1.0})
    with policy_proxy(POLICY_YAML) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "system", "content": "hi"}]},
        )

    assert calls == []
    assert response.json() == upstream_completion()


def test_policy_added_after_boot_is_enforced(make_proxy, checked):
    """`app.state.policies.add` takes effect on the next request, with no restart."""
    checked({DOOR_STATUS: 1.0})
    registry = PolicyRegistry()
    client = make_proxy(
        lambda request: DECLINE, response_hook=policy_guard, policies=registry
    )
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    with client:
        assert client.app.state.policies is registry
        before = client.post("/v1/chat/completions", json=request)
        client.app.state.policies.add(POLICY_YAML)
        after = client.post("/v1/chat/completions", json=request)

    assert before.json() == upstream_completion()
    assert after.json()["choices"][0]["message"]["content"] == REPLACEMENT


def test_policy_removed_after_boot_stops_being_enforced(policy_proxy, checked):
    """Removing a policy by its `risk_group_id` stops it screening replies."""
    checked({DOOR_STATUS: 1.0})
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    with policy_proxy(POLICY_YAML) as client:
        before = client.post("/v1/chat/completions", json=request)
        assert client.app.state.policies.remove("42") is True
        after = client.post("/v1/chat/completions", json=request)

    assert before.json()["choices"][0]["message"]["content"] == REPLACEMENT
    assert after.json() == upstream_completion()


def test_build_app_exposes_a_registry_by_default(make_proxy):
    """`app.state.policies` is always there, so callers need not pass one in."""
    client = make_proxy(lambda request: DECLINE)
    assert isinstance(client.app.state.policies, PolicyRegistry)
    assert client.app.state.policies.list() == []


def test_response_hook_sees_a_streamed_reply_assembled(make_proxy):
    """An upstream that streams is buffered and reassembled before the hook sees it."""
    seen = []

    def hook(request, reply):
        seen.append(reply)
        return DECLINE

    with make_proxy(lambda request: DECLINE, response_hook=hook) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert len(seen) == 1
    assert seen[0].choices[0].message.content == UPSTREAM_TEXT
    # Declining relays the upstream's own frames, so the client still gets its stream.
    frames = sse_payloads(response.text)
    assert "".join(f["choices"][0]["delta"]["content"] for f in frames) == UPSTREAM_TEXT


def test_response_hook_receives_the_request_body(make_proxy):
    """The hook can see what was asked, not just what came back."""
    seen = []

    with make_proxy(
        lambda request: DECLINE,
        response_hook=lambda request, reply: seen.append(request) or DECLINE,
    ) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert seen[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_response_hook_can_replace_a_reply_for_a_nonstreaming_client(make_proxy):
    """A hook returning a completion answers a non-streaming client directly."""
    with make_proxy(
        lambda request: DECLINE,
        response_hook=lambda request, reply: completion_from_text("screened", "m"),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.json()["choices"][0]["message"]["content"] == "screened"


def test_response_hook_exception_relays_the_upstream_reply(make_proxy):
    """A raising response hook must not cost the client its answer."""

    def hook(request, reply):
        raise RuntimeError("boom")

    with make_proxy(lambda request: DECLINE, response_hook=hook) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.json() == upstream_completion()


def test_response_hook_not_consulted_for_a_request_hook_reply(
    make_proxy, upstream_calls
):
    """A reply the request hook invented never came from the upstream, so it is not screened."""
    seen = []

    with make_proxy(
        lambda request: completion_from_text("from the request hook", "m"),
        response_hook=lambda request, reply: seen.append(reply) or DECLINE,
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert seen == []
    assert upstream_calls == []
    assert (
        response.json()["choices"][0]["message"]["content"] == "from the request hook"
    )


def test_upstream_error_is_relayed_unscreened(make_proxy):
    """An upstream error reaches the client with its status, and the hook never runs."""
    seen = []

    with make_proxy(
        lambda request: DECLINE,
        response_hook=lambda request, reply: seen.append(reply) or DECLINE,
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "upstream_error": True,
            },
        )

    assert seen == []
    assert response.status_code == 503
    assert response.json()["error"]["message"] == "overloaded"


def test_non_completion_reply_is_relayed_unscreened(make_proxy):
    """A 200 that is not a chat completion is relayed rather than shown to the hook."""
    seen = []

    with make_proxy(
        lambda request: DECLINE,
        response_hook=lambda request, reply: seen.append(reply) or DECLINE,
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "upstream_not_a_completion": True,
            },
        )

    assert seen == []
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def _mitm_cli() -> typer.Typer:
    """Wrap the `m mitm` command in its own app, so the CLI wiring can be invoked."""
    cli = typer.Typer()
    cli.command()(mitm)
    return cli


def test_run_server_seeds_policies_before_binding(tmp_path, monkeypatch):
    """Policies are loaded, and reachable on the app, before uvicorn binds a socket."""
    (tmp_path / "doors.yaml").write_text(POLICY_YAML)
    served: dict[str, Any] = {}

    monkeypatch.setattr(
        "cli.mitm.app.uvicorn.run",
        lambda app, host, port: served.update(app=app, host=host, port=port),
    )
    run_server(
        "http://upstream",
        port=9999,
        response_hook="cli.mitm.hooks:policy_guard",
        policies=[tmp_path / "doors.yaml"],
    )

    assert served["port"] == 9999
    assert [p.risk_group for p in served["app"].state.policies.list()] == [
        "pod_bay_doors"
    ]


def test_run_server_rejects_a_bad_policy_before_binding(tmp_path, monkeypatch):
    """A broken policy fails the command, rather than surfacing on some later request."""
    (tmp_path / "broken.yaml").write_text("just a string")
    monkeypatch.setattr(
        "cli.mitm.app.uvicorn.run",
        lambda *a, **k: pytest.fail("server bound despite an unloadable policy"),
    )

    with pytest.raises(PolicyError):
        run_server("http://upstream", policies=[tmp_path / "broken.yaml"])


def test_policy_option_implies_the_policy_guard(tmp_path, monkeypatch):
    """`--policy` on its own is enough; the response hook does not have to be named."""
    (tmp_path / "doors.yaml").write_text(POLICY_YAML)
    passed: dict[str, Any] = {}
    monkeypatch.setattr("cli.mitm.app.run_server", lambda **kw: passed.update(kw))

    result = CliRunner().invoke(
        _mitm_cli(),
        ["--upstream", "http://upstream", "--policy", str(tmp_path / "doors.yaml")],
    )

    assert result.exit_code == 0, result.output
    assert passed["response_hook"] == POLICY_HOOK
    assert passed["policies"] == [tmp_path / "doors.yaml"]
    # And the coin-flip default steps aside: a policy proxy that randomly refused half
    # its traffic would look exactly like a broken policy.
    assert passed["hook"] == PASSTHROUGH_HOOK


def test_response_hook_alone_also_silences_the_coin_flip(monkeypatch):
    """Screening replies is enough to turn off request interception."""
    passed: dict[str, Any] = {}
    monkeypatch.setattr("cli.mitm.app.run_server", lambda **kw: passed.update(kw))

    result = CliRunner().invoke(
        _mitm_cli(),
        ["--upstream", "http://upstream", "--response-hook", "my_mod:my_hook"],
    )

    assert result.exit_code == 0, result.output
    assert passed["hook"] == PASSTHROUGH_HOOK


def test_an_explicit_hook_still_wins(tmp_path, monkeypatch):
    """Naming `--hook` alongside `--policy` keeps the named hook."""
    (tmp_path / "doors.yaml").write_text(POLICY_YAML)
    passed: dict[str, Any] = {}
    monkeypatch.setattr("cli.mitm.app.run_server", lambda **kw: passed.update(kw))

    result = CliRunner().invoke(
        _mitm_cli(),
        [
            "--upstream",
            "http://upstream",
            "--policy",
            str(tmp_path / "doors.yaml"),
            "--hook",
            COIN_FLIP_HOOK,
        ],
    )

    assert result.exit_code == 0, result.output
    assert passed["hook"] == COIN_FLIP_HOOK


def test_no_policy_means_no_response_hook(monkeypatch):
    """Without `--policy`, nothing is screened and streaming stays incremental."""
    passed: dict[str, Any] = {}
    monkeypatch.setattr("cli.mitm.app.run_server", lambda **kw: passed.update(kw))

    result = CliRunner().invoke(_mitm_cli(), ["--upstream", "http://upstream"])

    assert result.exit_code == 0, result.output
    assert passed["response_hook"] is None
    assert passed["hook"] == COIN_FLIP_HOOK


def _sse_body(newline: str, *pieces: str, done: bool = True) -> bytes:
    """Frame content pieces as an SSE body, using the given line ending."""
    frames = [
        json.dumps(
            {
                "id": "chatcmpl-upstream",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "upstream-model",
                "choices": [
                    {"index": 0, "delta": {"content": p}, "finish_reason": None}
                ],
            }
        )
        for p in pieces
    ]
    if done:
        frames.append("[DONE]")
    return "".join(f"data: {f}{newline}{newline}" for f in frames).encode()


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_buffered_sse_assembles_under_either_framing(newline):
    """Reassembly must not depend on which line ending the upstream framed with."""
    body = _sse_body(newline, "The pod ", "bay doors ", "are open.")

    completion = asyncio.run(_completion_from_body(body, "text/event-stream"))

    assert completion is not None
    assert completion.choices[0].message.content == UPSTREAM_TEXT


def test_buffered_sse_skips_frames_that_are_not_chunks():
    """One malformed frame must not cost the client the rest of the reply."""
    body = b"data: {not json}\n\n: a comment line\n\n" + _sse_body(
        "\n", "The pod ", "bay doors ", "are open."
    )

    completion = asyncio.run(_completion_from_body(body, "text/event-stream"))

    assert completion is not None
    assert completion.choices[0].message.content == UPSTREAM_TEXT


def test_sse_body_is_detected_without_a_content_type():
    """An upstream that streams without labelling it is still screenable."""
    body = _sse_body("\n", "hello")

    completion = asyncio.run(_completion_from_body(body, ""))

    assert completion is not None
    assert completion.choices[0].message.content == "hello"


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
