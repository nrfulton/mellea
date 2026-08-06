# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardian shim: OpenAI-compatible server with policy-based guardrails.

Wraps an upstream OpenAI-compatible endpoint and checks all requests against
configurable policies using Granite Guardian adapters. Requests that violate
policies are politely declined before reaching the upstream model.
"""

import time
import uuid
from pathlib import Path

try:
    import httpx
    import typer
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError as e:
    raise ImportError(
        "The 'm guardian' command requires extra dependencies. "
        'Please install them with: pip install "mellea[server]"'
    ) from e

from typing import cast

import mellea
from cli.serve.models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionRequest,
    Choice,
    OpenAIError,
    OpenAIErrorResponse,
)
from mellea.backends.adapters import AdapterMixin
from mellea.core import MelleaLogger
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic.guardian import (
    guardian_check,
    policy_guardrails,
)
from mellea.stdlib.context import ChatContext

from .policy import GuardianPolicy, load_policy

logger = MelleaLogger.get_logger()

app = FastAPI(
    title="Guardian Shim - Policy-Based Guardrails Server",
    description="OpenAI-compatible proxy with Granite Guardian policy checking",
    version="0.1.0",
)

# Module-level state (set during run_server)
_policy: GuardianPolicy | None = None
_upstream_url: str | None = None
_guardian_model_id: str | None = None


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic liveness check endpoint."""
    return {"status": "pass"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert FastAPI validation errors to OpenAI-compatible format."""
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        param = first_error["loc"][-1] if first_error["loc"] else None
        message = first_error["msg"]
    else:
        param = None
        message = "Invalid request parameters"

    return _create_openai_error_response(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        param=str(param) if param else None,
    )


def _create_openai_error_response(
    status_code: int, message: str, error_type: str, param: str | None = None
) -> JSONResponse:
    """Create an OpenAI-compatible error response."""
    error_response = OpenAIErrorResponse(
        error=OpenAIError(message=message, type=error_type, param=param)
    )
    return JSONResponse(
        status_code=status_code, content=error_response.model_dump(mode="json")
    )


def _create_decline_response(
    decline_message: str, model: str, completion_id: str, created: int
) -> ChatCompletion:
    """Create a polite decline response."""
    return ChatCompletion(
        id=completion_id,
        model=model,
        created=created,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    content=decline_message, role="assistant", refusal=decline_message
                ),
                finish_reason="content_filter",
            )
        ],
        object="chat.completion",
        system_fingerprint=None,
        usage=None,
    )


def _build_context_from_messages(messages: list) -> ChatContext:
    """Convert OpenAI chat messages to a ChatContext for guardian checks."""
    context = ChatContext()
    for msg in messages:
        role = msg.role
        # Handle content that may be a string or list of content parts
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            # Extract text content from content parts
            text_parts = []
            for part in msg.content:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                elif hasattr(part, "root") and isinstance(part.root, str):
                    text_parts.append(part.root)
            content = " ".join(text_parts)
        else:
            content = str(msg.content) if msg.content else ""

        if role in ("user", "assistant", "system"):
            context = context.add(Message(role, content))
    return context


async def _check_policies(
    context: ChatContext, policy: GuardianPolicy, guardian_model_id: str
) -> str | None:
    """Check all policies against the context.

    Returns the decline message if any policy is violated, None otherwise.
    """
    session = mellea.start_session(model_id=guardian_model_id)
    # Guardian adapters require AdapterMixin; Granite models provide this
    backend = cast(AdapterMixin, session.backend)

    # Check policy rules (policy-guardrails adapter)
    for rule in policy.policy_rules:
        try:
            result = policy_guardrails(
                context, backend, rule.policy, model_options={"temperature": 0}
            )
            # "Yes" means compliant, "No" or "Ambiguous" means violation
            if result != "Yes":
                logger.info(f"Policy rule '{rule.name}' violated (result={result})")
                return rule.decline_message or policy.default_decline_message
        except Exception as e:
            logger.warning(f"Policy rule '{rule.name}' check failed: {e}")
            # On error, allow the request through (fail-open for availability)
            continue

    # Check guardian criteria (guardian-core adapter)
    for criteria in policy.guardian_criteria:
        try:
            score = guardian_check(
                context,
                backend,
                criteria.name,
                scoring_schema=criteria.scoring_schema,
                model_options={"temperature": 0},
            )
            if score >= criteria.threshold:
                logger.info(
                    f"Guardian criteria '{criteria.name}' triggered "
                    f"(score={score:.3f} >= threshold={criteria.threshold})"
                )
                return criteria.decline_message or policy.default_decline_message
        except Exception as e:
            logger.warning(f"Guardian criteria '{criteria.name}' check failed: {e}")
            continue

    return None


async def chat_completions_endpoint(
    request: ChatCompletionRequest,
) -> ChatCompletion | JSONResponse:
    """Handle chat completion requests with guardian policy checking."""
    global _policy, _upstream_url, _guardian_model_id

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_timestamp = int(time.time())

    # Validate n=1
    if request.n is not None and request.n > 1:
        return _create_openai_error_response(
            status_code=400,
            message=f"Multiple completions (n={request.n}) are not supported.",
            error_type="invalid_request_error",
            param="n",
        )

    # Build context for guardian checks
    context = _build_context_from_messages(request.messages)

    # Check policies
    if _policy is not None and _guardian_model_id is not None:
        decline_message = await _check_policies(context, _policy, _guardian_model_id)
        if decline_message:
            return _create_decline_response(
                decline_message, request.model, completion_id, created_timestamp
            )

    # Forward to upstream if we have one configured
    if _upstream_url:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{_upstream_url}/v1/chat/completions",
                json=request.model_dump(exclude_none=True, mode="json"),
            )
            if response.status_code == 200:
                return ChatCompletion.model_validate(response.json())
            else:
                return _create_openai_error_response(
                    status_code=response.status_code,
                    message=f"Upstream error: {response.text}",
                    error_type="server_error",
                )

    # No upstream - return error
    return _create_openai_error_response(
        status_code=500,
        message="No upstream server configured",
        error_type="server_error",
    )


def run_server(
    policy_path: str,
    upstream_url: str,
    guardian_model_id: str,
    host: str = "0.0.0.0",
    port: int = 8081,
) -> None:
    """Start the guardian shim server.

    Args:
        policy_path: Path to the YAML policy configuration file.
        upstream_url: URL of the upstream OpenAI-compatible endpoint to proxy to.
        guardian_model_id: Model ID for the guardian adapter checks.
        host: Host to bind to.
        port: Port to bind to.
    """
    global _policy, _upstream_url, _guardian_model_id

    # Load and validate policy
    policy_file = Path(policy_path)
    if not policy_file.exists():
        raise typer.BadParameter(f"Policy file not found: {policy_path}")

    _policy = load_policy(policy_path)
    _upstream_url = upstream_url.rstrip("/")
    _guardian_model_id = guardian_model_id

    typer.echo(f"Loaded policy from: {policy_path}")
    typer.echo(f"  - {len(_policy.policy_rules)} policy rules")
    typer.echo(f"  - {len(_policy.guardian_criteria)} guardian criteria")
    typer.echo(f"Guardian model: {guardian_model_id}")
    typer.echo(f"Upstream: {upstream_url}")

    # Add the endpoint
    app.add_api_route(
        "/v1/chat/completions",
        chat_completions_endpoint,
        methods=["POST"],
        response_model=ChatCompletion | OpenAIErrorResponse,
    )

    typer.echo(f"Starting guardian shim at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
