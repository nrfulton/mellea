# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer command definition for `m guardian`.

Separates the CLI interface (typer annotations) from the server implementation
(FastAPI, uvicorn) so that `m --help` works without the `server` extra installed.
"""

import typer

# Use string for typer default display
_DEFAULT_GUARDIAN_MODEL = "ibm-granite/granite-4.1-3b"


def guardian(
    policy_path: str = typer.Argument(
        ..., help="Path to the YAML policy configuration file"
    ),
    upstream_url: str = typer.Option(
        ...,
        "--upstream",
        "-u",
        help="URL of the upstream OpenAI-compatible endpoint to proxy to",
    ),
    guardian_model_id: str = typer.Option(
        _DEFAULT_GUARDIAN_MODEL,
        "--model",
        "-m",
        help="Model ID for the guardian adapter checks (must support Guardian adapters)",
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8081, help="Port to bind to"),
) -> None:
    """Run an OpenAI-compatible proxy with policy-based guardrails.

    The guardian shim sits between clients and an upstream LLM endpoint,
    checking all incoming requests against configurable policies using
    Granite Guardian adapters. Requests that violate policies are politely
    declined before reaching the upstream model.

    Policy files are YAML with two types of checks:

    1. `policy_rules`: Text policies checked with the policy-guardrails adapter.
       Returns "Yes" (compliant), "No", or "Ambiguous".

    2. `guardian_criteria`: Risk scoring with guardian-core adapter.
       Returns a 0.0-1.0 score; requests above threshold are declined.

    Prerequisites:
        Mellea installed with server dependency group (`uv add 'mellea[server]'`).
        An upstream OpenAI-compatible endpoint (e.g., `m serve`, vLLM, Ollama).
        A Granite model with Guardian adapter support.

    Output:
        Starts a long-running HTTP server on the specified host and port.
        The `/v1/chat/completions` endpoint proxies requests after policy checks.

    Examples:
        m guardian policy.yaml --upstream http://localhost:8080

        m guardian policy.yaml -u http://localhost:8080 --model ibm-granite/granite-4.1-8b

    See Also:
        guide: integrations/m-serve
    """
    from cli.guardian.app import run_server

    run_server(
        policy_path=policy_path,
        upstream_url=upstream_url,
        guardian_model_id=guardian_model_id,
        host=host,
        port=port,
    )
