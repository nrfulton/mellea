# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP control plane for the policies a `m mitm` proxy is enforcing.

`cli.mitm.policy.PolicyRegistry` is mutable while the proxy runs, and `policy_guard`
re-reads it for every reply it screens, but nothing reaches it from outside the process:
`cli.mitm.app.build_app` forwards every path it does not handle to the upstream server.
This module is that missing surface -- list, create, edit, enable, disable, and delete,
taking effect on the next screened reply with no restart.

It is deliberately **opt-in**. A proxy built without `admin=True` keeps forwarding
`/_mitm/*` upstream like any other path, so registering the router is the one thing that
makes the proxy less than perfectly transparent, and nothing about an existing deployment
changes until someone asks for it.

Requests carry a policy as JSON rather than YAML, because the editor driving this is a
form rather than a text box. Documents are still validated by `policy_from_mapping`, the
same function the YAML path ends up in, so a malformed policy is rejected here with the
message it would have been rejected with on the command line.

Security: this endpoint *removes safety guardrails*, and `m mitm` binds all interfaces by
default. Pass a token so that only holders of it can call these routes, and prefer binding
the proxy to a loopback interface when the control plane is on.
"""

import secrets
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from mellea.core.utils import MelleaLogger

from .policy import (
    Policy,
    PolicyEntry,
    PolicyError,
    PolicyRegistry,
    policy_from_mapping,
    policy_to_mapping,
)

logger = MelleaLogger.get_logger()

__all__ = ["ADMIN_PREFIX", "EnabledBody", "PolicyBody", "build_policy_router"]

ADMIN_PREFIX = "/_mitm"
"""Path prefix for the control plane.

Chosen to start with an underscore because no OpenAI-compatible route does, so mounting
this cannot shadow a path a client expects the upstream to answer.
"""


class PolicyBody(BaseModel):
    """A policy to register, as sent to `POST` and `PUT`.

    Args:
        policy: The policy document as a mapping in the `granite.trust.policy-tools`
            schema -- exactly what `policy_to_mapping` returns, so a document read from
            this API can be edited and sent straight back.
        enabled: Whether to enforce the policy. Omit to keep the flag the policy already
            has, or to enable a policy being created.
    """

    policy: dict[str, Any] = Field(
        description="Policy document in the granite.trust.policy-tools schema."
    )
    enabled: bool | None = Field(
        default=None, description="Whether to enforce it; omit to leave unchanged."
    )


class EnabledBody(BaseModel):
    """Whether a policy should be enforced, as sent to `PATCH`.

    Args:
        enabled: `True` to enforce the policy, `False` to park it without deleting it.
    """

    enabled: bool = Field(description="Whether to enforce the policy.")


def _entry_json(entry: PolicyEntry) -> dict[str, Any]:
    """Render a registry entry for the wire.

    The document and its enablement stay separate rather than being merged into one
    object, so what this returns under `policy` is a clean schema document and the
    response is shaped like the request body that would recreate it.

    Args:
        entry: The registry entry to render.

    Returns:
        A mapping with the policy document and whether it is being enforced.
    """
    return {"policy": policy_to_mapping(entry.policy), "enabled": entry.enabled}


def _parse(raw: Any) -> Policy:
    """Validate a policy document from a request body.

    Args:
        raw: The mapping the client sent.

    Returns:
        The parsed policy.

    Raises:
        HTTPException: 422 carrying the parser's own message, so a client sees which
            field it got wrong rather than a generic rejection.
    """
    try:
        return policy_from_mapping(raw)
    except PolicyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _find(registry: PolicyRegistry, name: str) -> PolicyEntry | None:
    """Look up a policy by `risk_group` name only.

    Deliberately narrower than `PolicyRegistry.get`, which also matches a
    `risk_group_id`: that is the right behaviour for addressing a policy in a URL, but the
    wrong test for whether a name is taken, because two groups may carry the same id.

    Args:
        registry: The registry to search.
        name: The `risk_group` name to look for.

    Returns:
        The matching entry, or `None`.
    """
    for entry in registry.entries():
        if entry.policy.risk_group == name:
            return entry
    return None


def _require(registry: PolicyRegistry, key: str) -> PolicyEntry:
    """Resolve the policy a request addressed.

    Args:
        registry: The registry to search.
        key: The policy's `risk_group` name or its `risk_group_id`.

    Returns:
        The matching entry.

    Raises:
        HTTPException: 404 if no policy matched.
    """
    entry = registry.get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No policy named {key!r}.")
    return entry


def _token_guard(token: str) -> Callable[..., None]:
    """Build the dependency that checks the admin bearer token.

    Args:
        token: The expected token.

    Returns:
        A FastAPI dependency that rejects any request not carrying it.
    """

    def guard(authorization: Annotated[str | None, Header()] = None) -> None:
        """Reject a request whose `Authorization` header does not carry the token.

        Raises:
            HTTPException: 401 if the header is missing, malformed, or does not match.
        """
        scheme, _, supplied = (authorization or "").partition(" ")
        # compare_digest over the whole comparison, so a wrong scheme and a wrong token
        # take the same time and neither leaks the token's length.
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            supplied.strip(), token
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing admin token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return guard


def build_policy_router(
    registry: PolicyRegistry, *, token: str | None = None
) -> APIRouter:
    """Build the policy control plane for a registry.

    Routes are mounted under `ADMIN_PREFIX` and act on the registry in place, so a change
    is live for the next reply `policy_guard` screens.

    Read-modify-write across two requests is not serialised: the registry locks each
    operation, not a sequence of them, so two clients editing the same policy at once
    resolve last-write-wins. That is the right trade for a single-operator control plane
    and the reason there is no revision or `ETag` here.

    Args:
        registry: The registry to expose. Normally the one on `app.state.policies`.
        token: Bearer token every request must carry, or `None` to require no
            authentication. `None` is only appropriate on a trusted interface, because
            these routes can delete a guardrail.

    Returns:
        A router providing list, read, create, replace, enable/disable, and delete.
    """
    router = APIRouter(
        prefix=ADMIN_PREFIX,
        tags=["policies"],
        dependencies=[Depends(_token_guard(token))] if token else [],
    )

    @router.get("/policies")
    async def list_policies() -> dict[str, Any]:
        """List every registered policy, enforced or parked, in registration order."""
        return {"policies": [_entry_json(entry) for entry in registry.entries()]}

    @router.get("/policies/{key}")
    async def get_policy(key: str) -> dict[str, Any]:
        """Read one policy, addressed by `risk_group` name or `risk_group_id`."""
        return _entry_json(_require(registry, key))

    @router.post("/policies", status_code=201)
    async def create_policy(body: PolicyBody) -> dict[str, Any]:
        """Register a new policy, refusing to overwrite one that already exists."""
        policy = _parse(body.policy)
        if _find(registry, policy.risk_group) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"A policy named {policy.risk_group!r} is already registered.",
            )
        registry.add_mapping(body.policy, enabled=body.enabled)
        logger.info("Registered policy %s via the control plane", policy.risk_group)
        return _entry_json(_require(registry, policy.risk_group))

    @router.put("/policies/{key}")
    async def replace_policy(key: str, body: PolicyBody) -> dict[str, Any]:
        """Replace a policy, optionally renaming its risk group.

        Renaming moves the policy to the end of the enforcement order, because the
        registry keys on `risk_group` and the old entry has to be dropped for the new
        name to take its place.
        """
        existing = _require(registry, key)
        policy = _parse(body.policy)
        renamed = policy.risk_group != existing.policy.risk_group

        if renamed and _find(registry, policy.risk_group) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"A policy named {policy.risk_group!r} is already registered.",
            )

        # Under a rename the registry sees a first-time registration and would default to
        # enabled, so carry the flag across explicitly.
        enabled = body.enabled
        if renamed:
            registry.remove(existing.policy.risk_group)
            if enabled is None:
                enabled = existing.enabled

        registry.add_mapping(body.policy, enabled=enabled)
        logger.info("Replaced policy %s via the control plane", policy.risk_group)
        return _entry_json(_require(registry, policy.risk_group))

    @router.patch("/policies/{key}")
    async def set_policy_enabled(key: str, body: EnabledBody) -> dict[str, Any]:
        """Start or stop enforcing a policy without deleting it."""
        entry = _require(registry, key)
        registry.set_enabled(entry.policy.risk_group, body.enabled)
        logger.info(
            "Policy %s is now %s",
            entry.policy.risk_group,
            "enforced" if body.enabled else "parked",
        )
        return _entry_json(_require(registry, entry.policy.risk_group))

    @router.delete("/policies/{key}", status_code=204)
    async def delete_policy(key: str) -> Response:
        """Stop enforcing a policy and forget it."""
        if not registry.remove(key):
            raise HTTPException(status_code=404, detail=f"No policy named {key!r}.")
        logger.info("Removed policy %r via the control plane", key)
        return Response(status_code=204)

    return router
