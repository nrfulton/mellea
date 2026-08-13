# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer command definition for `m mitm`.

Separates the CLI interface (typer annotations) from the proxy implementation
(FastAPI, httpx, uvicorn) so that `m --help` works without the `server` extra
installed. The heavy dependencies are only imported when `m mitm` is invoked.
"""

import typer


def mitm(
    upstream: str = typer.Option(
        ...,
        help="Base URL of the OpenAI-compatible server to front, e.g. http://localhost:11434",
    ),
    hook: str = typer.Option(
        "cli.mitm.hooks:coin_flip",
        help="Interceptor to run, as 'module:function' or 'path/to/file.py:function'",
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8081, help="Port to bind to"),
    timeout: float = typer.Option(
        600.0, help="Read timeout in seconds for upstream requests"
    ),
):
    """Proxy an OpenAI-compatible endpoint, intercepting requests with a hook.

    Starts a server that behaves exactly like `--upstream` unless a hook says
    otherwise. Every `POST /v1/chat/completions` request is passed to the hook, which
    returns either `None` to forward the request untouched, or a response
    (`openai.types.chat.ChatCompletion`, or an async iterator of
    `ChatCompletionChunk`) to answer it without contacting the upstream. All other
    paths are forwarded verbatim, so `/v1/models` and friends keep working.

    The hook does not need to handle streaming: whichever form it returns is adapted
    to the `stream` flag the client sent. The default hook refuses roughly half of all
    requests with a canned message, as a worked example of the contract.

    Prerequisites:
        Mellea installed with server dependency group (`uv add 'mellea[server]'`).
        A running OpenAI-compatible server to front.

    Output:
        Starts a long-running HTTP proxy on the specified host and port. Point any
        OpenAI client at `http://<host>:<port>/v1` in place of the upstream.

    Examples:
        m mitm --upstream http://localhost:11434

        m mitm --upstream http://localhost:8000 --hook my_checks.py:block_pii --port 9000

    See Also:
        guide: integrations/m-serve
    """
    from cli.mitm.app import run_server

    run_server(upstream=upstream, hook=hook, host=host, port=port, timeout=timeout)
