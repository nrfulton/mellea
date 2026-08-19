# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer command definition for `m mitm`.

Separates the CLI interface (typer annotations) from the proxy implementation
(FastAPI, httpx, uvicorn) so that `m --help` works without the `server` extra
installed. The heavy dependencies are only imported when `m mitm` is invoked.
"""

from pathlib import Path

import typer

POLICY_HOOK = "cli.mitm.hooks:policy_guard"
"""Response hook used when `--policy` is given without an explicit `--response-hook`."""

COIN_FLIP_HOOK = "cli.mitm.hooks:coin_flip"
"""Request hook used when nothing else is asked for: the worked example of the contract."""

PASSTHROUGH_HOOK = "cli.mitm.hooks:passthrough"
"""Request hook used when a reply is being screened, so requests are left alone."""


def mitm(
    upstream: str = typer.Option(
        ...,
        help="Base URL of the OpenAI-compatible server to front, e.g. http://localhost:11434",
    ),
    hook: str | None = typer.Option(
        None,
        help=(
            "Interceptor to run, as 'module:function' or 'path/to/file.py:function'. "
            "Defaults to no request interception when a policy or response hook is in "
            f"play, and to the {COIN_FLIP_HOOK} example otherwise"
        ),
    ),
    response_hook: str | None = typer.Option(
        None,
        help=(
            "Reply interceptor, as 'module:function'. Defaults to the policy guard when "
            "--policy is given, and to no reply interception otherwise"
        ),
    ),
    policy: list[Path] = typer.Option(
        [],
        help=(
            "Policy YAML file, or directory of them, to enforce. Repeatable. "
            "Uses the granite.trust.policy-tools schema"
        ),
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8081, help="Port to bind to"),
    timeout: float = typer.Option(
        600.0, help="Read timeout in seconds for upstream requests"
    ),
):
    """Proxy an OpenAI-compatible endpoint, intercepting requests and replies with hooks.

    Starts a server that behaves exactly like `--upstream` unless a hook says
    otherwise. Every `POST /v1/chat/completions` request is passed to the hook, which
    returns either `None` to forward the request untouched, or a response
    (`openai.types.chat.ChatCompletion`, or an async iterator of
    `ChatCompletionChunk`) to answer it without contacting the upstream. All other
    paths are forwarded verbatim, so `/v1/models` and friends keep working.

    A `--response-hook` is consulted for the reply the upstream produced, and may replace
    it. Use one to enforce anything that is a property of the reply rather than the
    request. Passing `--policy` is the ready-made case: each policy is a YAML document in
    the schema published by `ibm-granite/granite.trust.policy-tools`, and every
    restriction in its `reply_cannot_contain` lists is checked against the upstream's
    reply with the `requirement-check` adapter on `ibm-granite/granite-4.1-3b`. A reply
    that violates one is replaced by a refusal composed from that risk's
    `reply_may_contain` guidance.

    Neither hook needs to handle streaming: whichever form it returns is adapted to the
    `stream` flag the client sent. Note that a registered response hook has to buffer each
    upstream reply in full before it can judge it, so streamed replies stop arriving token
    by token; without one, streaming is relayed byte for byte as before.

    With no hook named and no policy given, the request hook refuses roughly half of all
    requests with a canned message, as a worked example of the contract. Screening replies
    turns that off: asking for `--policy` or `--response-hook` leaves the request side
    alone unless you name a `--hook` yourself.

    Prerequisites:
        Mellea installed with server dependency group (`uv add 'mellea[server]'`).
        A running OpenAI-compatible server to front.
        For `--policy`, the Hugging Face extra (`uv add 'mellea[hf]'`); the adapter
        weights are downloaded on the first screened reply.

    Output:
        Starts a long-running HTTP proxy on the specified host and port. Point any
        OpenAI client at `http://<host>:<port>/v1` in place of the upstream.

    Examples:
        m mitm --upstream http://localhost:11434

        m mitm --upstream http://localhost:8000 --hook my_checks.py:block_pii --port 9000

        m mitm --upstream http://localhost:11434 --policy policies/alcohol_prohibited.yaml

    See Also:
        guide: integrations/m-serve
    """
    from cli.mitm.app import run_server

    screening = response_hook or (POLICY_HOOK if policy else None)
    run_server(
        upstream=upstream,
        hook=hook or (PASSTHROUGH_HOOK if screening else COIN_FLIP_HOOK),
        host=host,
        port=port,
        timeout=timeout,
        response_hook=screening,
        policies=policy,
    )
