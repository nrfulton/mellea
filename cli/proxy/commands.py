# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer command definition for `m proxy`."""

import typer


def proxy(
    endpoint: str = typer.Option(
        ...,
        "--endpoint",
        help="Upstream OpenAI-compatible endpoint URL (e.g., http://127.0.0.1:7999/v1)",
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", help="Port to bind to"),
    mproxy: str | None = typer.Option(
        None,
        "--mproxy",
        help="Dotted path to MProxy implementation (e.g., mymodule.MyProxy)",
    ),
) -> None:
    """Proxy OpenAI chat requests to an upstream endpoint with optional rewriting.

    Starts a server that catches all OpenAI chat endpoint requests and proxies them
    verbatim to another OpenAI-compatible endpoint. Optionally apply request/response
    transformations via an MProxy implementation.

    Prerequisites:
        Mellea installed with server dependency group (`uv add 'mellea[server]'`).

    Output:
        Starts a long-running HTTP server on the specified host and port.
        The `/v1/chat/completions` endpoint proxies requests to the upstream endpoint.

    Examples:
        m proxy --endpoint="http://127.0.0.1:7999/v1" --host 0.0.0.0 --port 8080

        m proxy --endpoint="http://127.0.0.1:7999/v1" --mproxy cli.my_impl.MyProxy

    See Also:
        guide: integrations/m-proxy

    Args:
        endpoint: Upstream OpenAI-compatible endpoint URL.
        host: Host to bind to.
        port: Port to bind to.
        mproxy: Dotted path to an MProxy implementation class for request/response
            rewriting, or `None` to pass through unchanged.
    """
    from cli.proxy.mproxy import MProxy
    from cli.proxy.server import load_mproxy, run_proxy_server

    mproxy_instance: MProxy | None = None
    if mproxy:
        mproxy_instance = load_mproxy(mproxy)

    run_proxy_server(endpoint=endpoint, host=host, port=port, mproxy=mproxy_instance)
