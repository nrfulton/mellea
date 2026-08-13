# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hook contract and helpers for `m mitm`.

A hook inspects an OpenAI-compatible chat-completion request and decides what the
client should see. Returning `None` means "let the upstream server answer this" --
the proxy forwards the original request untouched. Returning a response means "answer
this myself" -- the proxy sends that response and never contacts the upstream.

The return type is deliberately the same type the proxy would have produced from the
upstream server, so there is one representation of a response rather than two:
`openai.types.chat.ChatCompletion` for a whole reply, or an async iterator of
`openai.types.chat.ChatCompletionChunk` for a streamed one.

A hook does not need to care whether the client asked for streaming. The proxy adapts
whichever form the hook returns to whichever form the client requested.
"""

import random
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeAlias

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

Reply: TypeAlias = ChatCompletion | AsyncIterator[ChatCompletionChunk]
"""What a hook may send back to the client instead of the upstream's answer."""

Hook: TypeAlias = Callable[[dict[str, Any]], Awaitable[Reply | None] | Reply | None]
"""A request interceptor.

Takes the parsed request body and returns a `Reply` to answer the request itself, or
`None` to let the upstream server answer it. May be a plain function or a coroutine
function; the proxy awaits the result if it is awaitable.
"""

__all__ = [
    "Hook",
    "Reply",
    "chunks_from_text",
    "coin_flip",
    "completion_from_text",
    "new_completion_id",
]

REFUSAL = "Sorry Dave, I can't do that"
"""The canned reply used by the `coin_flip` example hook."""


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
