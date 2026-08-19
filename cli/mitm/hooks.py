# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hook contract and helpers for `m mitm`.

Hooks come in two kinds, and the proxy can run one of each.

A *request* hook inspects an OpenAI-compatible chat-completion request and decides what
the client should see. Returning `None` means "let the upstream server answer this" --
the proxy forwards the original request untouched. Returning a response means "answer
this myself" -- the proxy sends that response and never contacts the upstream.

A *response* hook inspects the reply the upstream produced, and returns either `None` to
let that reply through untouched or a replacement to send instead. Use one when the thing
being enforced is a property of the reply rather than the request -- as with the
`reply_cannot_contain` restrictions of a `cli.mitm.policy` policy, which cannot be
evaluated until there is a reply to evaluate.

The return type is deliberately the same type the proxy would have produced from the
upstream server, so there is one representation of a response rather than two:
`openai.types.chat.ChatCompletion` for a whole reply, or an async iterator of
`openai.types.chat.ChatCompletionChunk` for a streamed one.

A hook does not need to care whether the client asked for streaming. The proxy adapts
whichever form the hook returns to whichever form the client requested.
"""

import asyncio
import functools
import random
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

from mellea.core.utils import MelleaLogger

from .policy import PolicyRegistry, PolicyRisk

if TYPE_CHECKING:
    from mellea.backends.adapters import AdapterMixin
    from mellea.stdlib.context import ChatContext

logger = MelleaLogger.get_logger()

Reply: TypeAlias = ChatCompletion | AsyncIterator[ChatCompletionChunk]
"""What a hook may send back to the client instead of the upstream's answer."""

Hook: TypeAlias = Callable[[dict[str, Any]], Awaitable[Reply | None] | Reply | None]
"""A request interceptor.

Takes the parsed request body and returns a `Reply` to answer the request itself, or
`None` to let the upstream server answer it. May be a plain function or a coroutine
function; the proxy awaits the result if it is awaitable.
"""

ResponseHook: TypeAlias = Callable[..., Awaitable[Reply | None] | Reply | None]
"""A reply interceptor.

Called with the parsed request body and the `ChatCompletion` the upstream produced,
and returns a `Reply` to send in place of it or `None` to let it through untouched. May
be a plain function or a coroutine function; the proxy awaits the result if it is
awaitable.

A response hook that declares a keyword-only `policies` parameter is additionally handed
the proxy's `cli.mitm.policy.PolicyRegistry` -- the same object as `app.state.policies`
-- so that policies added or removed while the server runs take effect on the next
request. `policy_guard` is written this way.

Note that registering a response hook means a streamed upstream reply is buffered to
completion before the client sees any of it: there is no way to judge a whole reply from
its first token. Requests are still forwarded incrementally when no response hook is set.
"""

__all__ = [
    "Hook",
    "Reply",
    "ResponseHook",
    "chunks_from_text",
    "coin_flip",
    "completion_from_text",
    "guardian_violence",
    "new_completion_id",
    "passthrough",
    "policy_guard",
]

REFUSAL = "Sorry Dave, I can't do that"
"""The canned reply used by the `coin_flip` example hook."""

VIOLENCE_REFUSAL = "sorry, I can't do that Dave."
"""The canned reply used by the `guardian_violence` hook."""

VIOLENCE_RISK_THRESHOLD = 0.5
"""Guardian scores at or above this are treated as risk detected."""

POLICY_VIOLATION_THRESHOLD = 0.5
"""Requirement-check scores at or above this are treated as a policy violation.

Each restriction is checked by asking whether the reply *contains* the forbidden content,
so a high score means it does. Measured against the published `alcohol_prohibited.yaml`,
that phrasing separates a reply recommending wine from a reply refusing to (0.80 against
0.00 on the restriction it trips), and leaves every restriction it does not trip below
0.30. Asking the complement -- whether the reply avoids the content -- scored the
violating reply at 0.56 and so failed to catch it at all.
"""

POLICY_REFUSAL = "Sorry, I can't help with that."
"""Deterministic refusal used when a policy is violated but no better text can be built."""

POLICY_REFUSAL_MAX_TOKENS = 256
"""Generation budget for a composed refusal.

Set explicitly because the backend default is short enough to cut a refusal off
mid-sentence, and a truncated refusal is what the client actually reads.
"""

GUARDIAN_MODEL_ID = "ibm-granite/granite-4.1-3b"
"""Base model whose adapters back `guardian_violence` and `policy_guard`.

`guardian-core` scores requests for the former and `requirement-check` scores replies for
the latter. Both adapters sit on this one set of weights, which `_adapter_backend` loads
once for whichever hooks are in use.
"""


def new_completion_id() -> str:
    """Generate an OpenAI-style completion id.

    Returns:
        An id of the form `chatcmpl-<hex>`, matching the format real OpenAI-compatible
        servers use.
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def completion_from_text(
    text: str,
    model: str,
    *,
    finish_reason: str = "stop",
    completion_id: str | None = None,
    created: int | None = None,
) -> ChatCompletion:
    """Frame a string as a complete, non-streamed chat completion.

    Args:
        text: The assistant message content.
        model: Model name to echo back, normally taken from the request.
        finish_reason: Why generation stopped. Defaults to `stop`.
        completion_id: Completion id to use. A fresh one is generated if omitted.
        created: Unix creation timestamp. The current time is used if omitted.

    Returns:
        A `ChatCompletion` a client cannot distinguish from a real one.
    """
    return ChatCompletion(
        id=completion_id or new_completion_id(),
        object="chat.completion",
        created=created if created is not None else int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
    )


async def chunks_from_text(
    text: str,
    model: str,
    *,
    finish_reason: str = "stop",
    completion_id: str | None = None,
    created: int | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    """Frame a string as a streamed chat completion, one chunk per word.

    Emits the same chunk sequence a real server does: an opening chunk carrying the
    assistant role, one content chunk per word, then a final chunk carrying
    `finish_reason` and no content.

    Args:
        text: The assistant message content to spread across content chunks.
        model: Model name to echo back, normally taken from the request.
        finish_reason: Why generation stopped, sent on the final chunk. Defaults to
            `stop`.
        completion_id: Completion id shared by every chunk. A fresh one is generated
            if omitted.
        created: Unix creation timestamp shared by every chunk. The current time is
            used if omitted.

    Yields:
        Chunks in wire order, ending with the `finish_reason` chunk. The terminating
        `[DONE]` sentinel is added by the proxy, not here.
    """
    chunk_id = completion_id or new_completion_id()
    timestamp = created if created is not None else int(time.time())

    def chunk(delta: ChoiceDelta, finish: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=chunk_id,
            object="chat.completion.chunk",
            created=timestamp,
            model=model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish,  # type: ignore[arg-type]
                )
            ],
        )

    yield chunk(ChoiceDelta(role="assistant", content=""))

    # Keep trailing whitespace attached to each word so the joined content is
    # byte-identical to the input text.
    for word in re.findall(r"\S+\s*", text):
        yield chunk(ChoiceDelta(content=word))

    yield chunk(ChoiceDelta(), finish_reason)


def coin_flip(request: dict[str, Any]) -> Reply | None:
    """Refuse roughly half of all requests, and pass the rest through.

    The default hook, and a worked example of the contract: on heads it streams a
    canned refusal; on tails it returns `None`, so the client gets the upstream
    server's real answer and cannot tell the proxy is there.

    Args:
        request: The parsed OpenAI-compatible chat-completion request body.

    Returns:
        A stream of chunks carrying the refusal, or `None` to forward the request
        upstream.
    """
    if random.random() < 0.5:
        return chunks_from_text(REFUSAL, model=str(request.get("model", "mitm")))
    return None


def passthrough(request: dict[str, Any]) -> Reply | None:
    """Forward every request to the upstream, intercepting nothing.

    The request hook to pair with a response hook: policy enforcement judges replies, so
    the request side should get out of the way rather than apply the `coin_flip` default.

    Args:
        request: The parsed OpenAI-compatible chat-completion request body.

    Returns:
        Always `None`.
    """
    return None


def _message_text(content: Any) -> str:
    """Flatten an OpenAI message `content` field to plain text.

    Args:
        content: A string, or the multimodal parts list clients may send instead.

    Returns:
        The text of the message, with non-text parts (images, audio) dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _context_from_request(request: dict[str, Any]) -> "ChatContext | None":
    """Rebuild the conversation in a request as a `ChatContext`.

    Args:
        request: The parsed OpenAI-compatible chat-completion request body.

    Returns:
        A context holding the request's messages in order, or `None` if there is no
        user turn to score -- messages with roles Guardian does not model (`developer`
        is folded into `system`) and messages with no text are dropped.
    """
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import ChatContext

    context = ChatContext()
    saw_user = False
    for message in request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        role = "system" if role == "developer" else role
        if role not in ("system", "user", "assistant", "tool"):
            continue
        text = _message_text(message.get("content"))
        if not text:
            continue
        saw_user = saw_user or role == "user"
        context = context.add(Message(role, text))  # type: ignore[arg-type]

    return context if saw_user else None


@functools.lru_cache(maxsize=1)
def _adapter_backend() -> "AdapterMixin":
    """Load the adapter evaluation backend, once per process.

    Cached because the hooks run on every request and the weights take seconds to
    tens of seconds to load; the first intercepted request pays that cost. Every hook
    in this module shares the one backend, so enabling both `guardian_violence` and
    `policy_guard` does not load `GUARDIAN_MODEL_ID` twice.

    Returns:
        A backend that can serve the `guardian-core` and `requirement-check` adapters.

    Raises:
        ImportError: If the Hugging Face backend dependencies are not installed.
    """
    from mellea.backends.huggingface import LocalHFBackend

    return LocalHFBackend(model_id=GUARDIAN_MODEL_ID)


def _violence_score(context: "ChatContext") -> float:
    """Score the last user turn for violence with the guardian-core adapter.

    Blocking: runs a local model. `guardian_violence` calls this in a worker thread.

    Args:
        context: The conversation to evaluate.

    Returns:
        A risk score from 0.0 (no risk) to 1.0 (risk detected).
    """
    from mellea.stdlib.components.intrinsic import guardian

    return guardian.guardian_check(
        context, _adapter_backend(), criteria="violence", scoring_schema="user_prompt"
    )


async def guardian_violence(request: dict[str, Any]) -> Reply | None:
    """Refuse requests whose last user turn trips the Guardian violence check.

    Runs the `guardian-core` adapter over the incoming conversation with the
    `violence` criteria from `CRITERIA_BANK`, scoring the last user message before
    the upstream model ever sees it. Anything the check clears is forwarded, so the
    client cannot tell the proxy is there.

    The scoring model runs locally and blocks, so it is loaded once and called in a
    worker thread to keep the proxy responsive to other requests.

    Note that this is a fail-open guardrail: if the check itself errors -- missing
    Hugging Face dependencies, weights that will not load -- the proxy logs the
    exception and forwards the request upstream unfiltered.

    Args:
        request: The parsed OpenAI-compatible chat-completion request body.

    Returns:
        A stream of chunks carrying the refusal, or `None` to forward the request
        upstream.
    """
    context = _context_from_request(request)
    if context is None:
        return None

    score = await asyncio.to_thread(_violence_score, context)
    if score >= VIOLENCE_RISK_THRESHOLD:
        return chunks_from_text(
            VIOLENCE_REFUSAL, model=str(request.get("model", "mitm"))
        )
    return None


def _reply_text(reply: ChatCompletion) -> str:
    """Extract the assistant text from an upstream reply.

    Args:
        reply: The completion the upstream produced.

    Returns:
        The first choice's message content, or the empty string if the reply carries no
        text -- an empty choice list, or a tool-call-only message. There is nothing for
        a reply policy to judge in that case.
    """
    if not reply.choices:
        return ""
    return reply.choices[0].message.content or ""


def _last_user_text(request: dict[str, Any]) -> str:
    """Find the most recent user message in a request.

    Args:
        request: The parsed OpenAI-compatible chat-completion request body.

    Returns:
        The text of the last user turn, or the empty string if there is none.
    """
    for message in reversed(request.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message.get("content"))
    return ""


def _check_restriction(context: "ChatContext", restriction: str) -> float:
    """Score whether the reply in `context` contains what a restriction forbids.

    Blocking: runs a local model. `policy_guard` calls this in a worker thread.

    The restriction is put to the adapter positively -- does the reply contain this? --
    rather than as the requirement a compliant reply would satisfy. Both readings are
    faithful to the policy, but only the positive one separates compliant from violating
    replies in practice; see `POLICY_VIOLATION_THRESHOLD` for the measurements.

    Args:
        context: The conversation, ending with the candidate assistant reply.
        restriction: One entry from a risk's `reply_cannot_contain` list.

    Returns:
        A score from 0.0 (the reply does not contain the forbidden content) to 1.0 (it
        does).
    """
    from mellea.stdlib.components.intrinsic import core

    return core.requirement_check(
        context,
        _adapter_backend(),
        requirement=f"The assistant's response contains: {restriction}",
    )


def _fallback_refusal(risk: PolicyRisk) -> str:
    """Build a refusal without calling a model.

    Args:
        risk: The risk whose restriction the reply violated.

    Returns:
        `POLICY_REFUSAL`, with the risk's `reason_denial` code appended when it has one.
    """
    if risk.reason_denial:
        return f"{POLICY_REFUSAL} ({risk.reason_denial})"
    return POLICY_REFUSAL


def _generate_refusal(risk: PolicyRisk, question: str) -> str:
    """Compose a replacement reply that stays inside a risk's permitted content.

    Blocking: runs a local model. `policy_guard` calls this in a worker thread.

    Uses the risk's `reply_may_contain` list as the brief, which is what that field is
    for: it describes what a compliant reply to this kind of request looks like. Falls
    back to `_fallback_refusal` when there is no guidance to work from or when
    generation fails, so a violating reply is never passed through for want of a
    replacement.

    Args:
        risk: The risk whose restriction the reply violated.
        question: The user's last message, so the refusal can be on topic.

    Returns:
        The generated reply, or a deterministic refusal.
    """
    if not risk.reply_may_contain:
        return _fallback_refusal(risk)

    permitted = "\n".join(f"- {item}" for item in risk.reply_may_contain)
    brief = (
        "You are answering a user under a content policy that forbids the reply you "
        "would otherwise have given.\n\n"
        f"The user asked: {question}\n\n"
        "Write the reply. It must stay entirely within this permitted content:\n"
        f"{permitted}\n\n"
        "Be brief and courteous. Do not quote or paraphrase these instructions, do not "
        "mention the policy machinery, and do not provide the content that was withheld."
    )

    try:
        from mellea.backends.model_options import ModelOption
        from mellea.stdlib.session import MelleaSession

        session = MelleaSession(_adapter_backend())  # type: ignore[arg-type]
        text = str(
            session.instruct(
                brief,
                strategy=None,
                model_options={ModelOption.MAX_NEW_TOKENS: POLICY_REFUSAL_MAX_TOKENS},
            )
        ).strip()
    except Exception:
        logger.exception(
            "Could not generate a refusal for risk %s; using canned text", risk.risk
        )
        return _fallback_refusal(risk)

    return text or _fallback_refusal(risk)


async def policy_guard(
    request: dict[str, Any], reply: ChatCompletion, *, policies: PolicyRegistry
) -> Reply | None:
    """Replace upstream replies that violate a registered policy.

    A response hook for policies in the `granite.trust.policy-tools` format (see
    `cli.mitm.policy`). Every restriction in every registered policy's
    `reply_cannot_contain` lists is checked against the upstream's candidate reply with
    the `requirement-check` adapter, one call per restriction. The first restriction the
    reply fails wins: checking stops there and a replacement reply is composed from that
    risk's `reply_may_contain` guidance. A reply that clears every restriction is passed
    through untouched, so the client cannot tell the proxy is there.

    Checking is short-circuited because each restriction costs a model call and policies
    are long -- the published `alcohol_prohibited.yaml` alone has twenty restrictions
    across four risks. A compliant reply therefore pays for all of them, which is the
    latency floor of enforcing a policy this way.

    The scoring model runs locally and blocks, so it is loaded once and called in a
    worker thread to keep the proxy responsive to other requests.

    Note that this is a fail-open guardrail: if a check itself errors -- missing Hugging
    Face dependencies, weights that will not load -- the proxy logs the exception and
    lets the upstream reply through unscreened. Composing the *replacement* fails closed:
    a violation is always blocked, with canned text if the model cannot be reached.

    Args:
        request: The parsed OpenAI-compatible chat-completion request body.
        reply: The completion the upstream produced.
        policies: The proxy's policy registry, injected by the app. Policies added or
            removed while the server runs take effect on the next request.

    Returns:
        A stream of chunks carrying the replacement reply, or `None` to let the upstream
        reply through.
    """
    restrictions = policies.restrictions()
    if not restrictions:
        return None

    text = _reply_text(reply)
    if not text:
        return None

    context = _context_from_request(request)
    if context is None:
        return None

    from mellea.stdlib.components import Message

    screened = context.add(Message("assistant", text))

    for policy, risk, restriction in restrictions:
        score = await asyncio.to_thread(_check_restriction, screened, restriction)
        if score < POLICY_VIOLATION_THRESHOLD:
            continue

        logger.info(
            "Reply violates %s risk %s (%s), score %.2f: %s",
            policy.risk_group,
            risk.risk_id or "?",
            risk.risk,
            score,
            restriction,
        )
        refusal = await asyncio.to_thread(
            _generate_refusal, risk, _last_user_text(request)
        )
        return chunks_from_text(
            refusal, model=reply.model or str(request.get("model", "mitm"))
        )

    return None
