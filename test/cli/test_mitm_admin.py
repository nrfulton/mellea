# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `m mitm` policy control plane.

No model and no real upstream. The control plane only parses and stores policies, so what
matters here is the routes, their status codes, the fact that mounting them is opt-in, and
that mounting them does not shadow a path the client expects the upstream to answer.
Enforcement of what they register is covered in `test_mitm.py`.

The upstream is a stub reached over `httpx.ASGITransport`, so the real proxy code path runs
without opening a socket -- the same arrangement `test_mitm.py` uses.
"""

from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cli.mitm.admin import ADMIN_PREFIX
from cli.mitm.app import build_app
from cli.mitm.hooks import passthrough

POLICIES = f"{ADMIN_PREFIX}/policies"

# A status no part of the proxy produces, so a reply carrying it can only have come from
# the stub upstream -- which is how the opt-in tests tell forwarding from handling.
UPSTREAM_STATUS = 418


def policy_document(
    name: str = "alcohol_consumption_prohibited",
    *,
    group_id: Any = 11,
    version: str = "v1.0",
    restrictions: tuple[str, ...] = ("Recommendations for alcoholic beverages",),
) -> dict[str, Any]:
    """Build a policy document in the shape a JSON client sends.

    Args:
        name: The `risk_group` name.
        group_id: The `risk_group_id`, as a number by default because that is what a JSON
            client sends and what YAML parses these as.
        version: The `policy_version`.
        restrictions: The risk's `reply_cannot_contain` entries.

    Returns:
        The document, ready to post.
    """
    return {
        "risk_group": name,
        "risk_group_id": group_id,
        "description": "Policy for jurisdictions where alcohol is prohibited.",
        "policy_version": version,
        "risks": [
            {
                "risk": "alcohol_general_requests",
                "risk_id": "11.1",
                "description": "Requests for information about alcohol",
                "reason_denial": "ALCOHOL_PROHIBITED",
                "policy": {
                    "reply_cannot_contain": list(restrictions),
                    "reply_may_contain": ["Polite refusal"],
                },
            }
        ],
    }


@pytest.fixture
def upstream_calls() -> list[str]:
    """Paths the stub upstream was asked for."""
    return []


@pytest.fixture
def upstream_app(upstream_calls: list[str]) -> FastAPI:
    """An upstream that answers anything, so forwarding is visible in the reply."""
    app = FastAPI()

    @app.api_route(
        "/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
    )
    async def anything(request: Request, full_path: str) -> Any:
        upstream_calls.append(request.url.path)
        return JSONResponse({"upstream": True}, status_code=UPSTREAM_STATUS)

    return app


@pytest.fixture
def make_admin(upstream_app: FastAPI):
    """Build a `TestClient` for a proxy with the control plane mounted.

    The registry is reachable as `client.app.state.policies`, which is the same object the
    routes mutate and `policy_guard` would read.
    """

    def _make(*, admin: bool = True, token: str | None = None) -> TestClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app), base_url="http://upstream"
        )
        return TestClient(
            build_app(
                "http://upstream",
                passthrough,
                admin=admin,
                admin_token=token,
                client=client,
            )
        )

    return _make


# --------------------------------------------------------------------------------------
# Mounting is opt-in
# --------------------------------------------------------------------------------------


def test_control_plane_is_off_by_default(make_admin, upstream_calls):
    """Without `admin`, the control-plane paths forward like any other path.

    This is what keeps the proxy transparent for deployments that never asked for a
    control plane: nothing about them changes because the routes exist in the codebase.
    """
    with make_admin(admin=False) as client:
        response = client.get(POLICIES)

    assert response.status_code == UPSTREAM_STATUS
    assert response.json() == {"upstream": True}
    assert upstream_calls == [POLICIES]


def test_mounting_does_not_capture_other_paths(make_admin, upstream_calls):
    """Only `/_mitm` is answered locally; everything else still reaches the upstream."""
    with make_admin() as client:
        models = client.get("/v1/models")
        completions = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert models.status_code == UPSTREAM_STATUS
    assert completions.status_code == UPSTREAM_STATUS
    assert upstream_calls == ["/v1/models", "/v1/chat/completions"]


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def test_list_is_empty_on_a_fresh_registry(make_admin):
    """A proxy started without policies reports none."""
    with make_admin() as client:
        response = client.get(POLICIES)

    assert response.status_code == 200
    assert response.json() == {"policies": []}


def test_list_reports_the_document_and_its_enablement(make_admin):
    """Each entry carries a clean schema document plus whether it is enforced."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        entry = client.get(POLICIES).json()["policies"][0]

    assert entry["enabled"] is True
    assert entry["policy"]["risk_group"] == "alcohol_consumption_prohibited"
    # The flag stays out of the document, so what is under `policy` is still a policy.
    assert "enabled" not in entry["policy"]


@pytest.mark.parametrize("key", ["alcohol_consumption_prohibited", "11"])
def test_get_addresses_a_policy_by_name_or_id(make_admin, key):
    """Either identifier a document carries works in the URL."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        response = client.get(f"{POLICIES}/{key}")

    assert response.status_code == 200
    assert response.json()["policy"]["risk_group_id"] == "11"


def test_get_reports_a_miss(make_admin):
    """An unknown policy is a 404 naming what was asked for."""
    with make_admin() as client:
        response = client.get(f"{POLICIES}/no_such_policy")

    assert response.status_code == 404
    assert "no_such_policy" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# Creating
# --------------------------------------------------------------------------------------


def test_create_registers_and_enforces_a_policy(make_admin):
    """A created policy is live: the registry the hook reads has its restrictions."""
    with make_admin() as client:
        response = client.post(POLICIES, json={"policy": policy_document()})
        registry = client.app.state.policies

    assert response.status_code == 201
    assert response.json()["enabled"] is True
    assert [r for _, _, r in registry.restrictions()] == [
        "Recommendations for alcoholic beverages"
    ]


def test_create_coerces_a_numeric_id_the_way_yaml_does(make_admin):
    """JSON numbers become strings, so ids match however the policy arrived."""
    with make_admin() as client:
        response = client.post(POLICIES, json={"policy": policy_document(group_id=11)})

    assert response.json()["policy"]["risk_group_id"] == "11"


def test_create_can_register_a_policy_parked(make_admin):
    """A policy can be staged without being enforced yet."""
    with make_admin() as client:
        response = client.post(
            POLICIES, json={"policy": policy_document(), "enabled": False}
        )
        registry = client.app.state.policies

    assert response.json()["enabled"] is False
    assert registry.restrictions() == []
    assert len(registry) == 1


def test_create_refuses_to_overwrite_an_existing_policy(make_admin):
    """A duplicate name is a 409, so a create cannot silently replace a guard."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        response = client.post(POLICIES, json={"policy": policy_document()})

    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]


def test_create_reports_the_parser_own_message(make_admin):
    """A malformed document is rejected with the message the CLI would have printed.

    The point of routing JSON through `policy_from_mapping` is that there is one schema
    implementation and one set of error messages, so this pins the message reaching the
    client rather than merely the status.
    """
    with make_admin() as client:
        response = client.post(POLICIES, json={"policy": {"description": "no name"}})

    assert response.status_code == 422
    assert response.json()["detail"] == "Policy is missing a 'risk_group' name."


def test_create_rejects_a_bad_risk_without_registering_anything(make_admin):
    """A document that fails halfway through leaves the registry untouched."""
    document = policy_document()
    document["risks"].append({"description": "a risk with no name"})

    with make_admin() as client:
        response = client.post(POLICIES, json={"policy": document})
        registry = client.app.state.policies

    assert response.status_code == 422
    assert "risks[1] is missing a 'risk' name" in response.json()["detail"]
    assert len(registry) == 0


# --------------------------------------------------------------------------------------
# Replacing
# --------------------------------------------------------------------------------------


def test_replace_updates_a_policy_in_place(make_admin):
    """A replaced policy takes over from the old one rather than joining it."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        response = client.put(
            f"{POLICIES}/alcohol_consumption_prohibited",
            json={
                "policy": policy_document(
                    version="v1.1", restrictions=("Recipes for alcoholic beverages",)
                )
            },
        )
        registry = client.app.state.policies

    assert response.status_code == 200
    assert response.json()["policy"]["policy_version"] == "v1.1"
    assert len(registry) == 1
    assert [r for _, _, r in registry.restrictions()] == [
        "Recipes for alcoholic beverages"
    ]


def test_replace_preserves_a_parked_policy(make_admin):
    """Editing a disabled guard does not re-arm it.

    A form that reads a policy, changes a restriction, and writes it back must not turn
    enforcement back on as a side effect of saving.
    """
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        client.patch(f"{POLICIES}/11", json={"enabled": False})

        response = client.put(f"{POLICIES}/11", json={"policy": policy_document()})
        registry = client.app.state.policies

    assert response.json()["enabled"] is False
    assert registry.restrictions() == []


def test_replace_can_set_enablement_explicitly(make_admin):
    """A save may also change the flag, so one request can edit and arm a policy."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document(), "enabled": False})
        response = client.put(
            f"{POLICIES}/11", json={"policy": policy_document(), "enabled": True}
        )
        registry = client.app.state.policies

    assert response.json()["enabled"] is True
    assert len(registry.restrictions()) == 1


def test_replace_can_rename_a_risk_group(make_admin):
    """Renaming moves the policy rather than forking it."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        response = client.put(
            f"{POLICIES}/alcohol_consumption_prohibited",
            json={"policy": policy_document("alcohol_restricted")},
        )
        listed = client.get(POLICIES).json()["policies"]
        registry = client.app.state.policies

    assert response.status_code == 200
    assert [e["policy"]["risk_group"] for e in listed] == ["alcohol_restricted"]
    assert len(registry) == 1


def test_rename_carries_the_parked_flag_across(make_admin):
    """A rename is a remove and an add, so the flag has to be carried deliberately."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document(), "enabled": False})
        response = client.put(
            f"{POLICIES}/alcohol_consumption_prohibited",
            json={"policy": policy_document("alcohol_restricted")},
        )
        registry = client.app.state.policies

    assert response.json()["enabled"] is False
    assert registry.restrictions() == []


def test_replace_refuses_a_rename_onto_another_policy(make_admin):
    """Renaming onto a name in use is a 409, and neither policy is touched."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        client.post(POLICIES, json={"policy": policy_document("competitor_statements")})

        response = client.put(
            f"{POLICIES}/alcohol_consumption_prohibited",
            json={"policy": policy_document("competitor_statements")},
        )
        listed = client.get(POLICIES).json()["policies"]

    assert response.status_code == 409
    assert [e["policy"]["risk_group"] for e in listed] == [
        "alcohol_consumption_prohibited",
        "competitor_statements",
    ]


def test_replace_reports_a_miss(make_admin):
    """`PUT` updates an existing policy; it does not create one."""
    with make_admin() as client:
        response = client.put(
            f"{POLICIES}/no_such_policy", json={"policy": policy_document()}
        )
        registry = client.app.state.policies

    assert response.status_code == 404
    assert len(registry) == 0


# --------------------------------------------------------------------------------------
# Enabling, disabling, deleting
# --------------------------------------------------------------------------------------


def test_patch_parks_and_restores_enforcement(make_admin):
    """The toggle is what stops and starts screening, without losing the policy."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})
        registry = client.app.state.policies

        parked = client.patch(f"{POLICIES}/11", json={"enabled": False})
        assert parked.json()["enabled"] is False
        assert registry.restrictions() == []
        assert len(registry) == 1

        armed = client.patch(f"{POLICIES}/11", json={"enabled": True})
        assert armed.json()["enabled"] is True
        assert len(registry.restrictions()) == 1


def test_patch_reports_a_miss(make_admin):
    """Toggling something absent is a 404."""
    with make_admin() as client:
        response = client.patch(f"{POLICIES}/no_such_policy", json={"enabled": False})

    assert response.status_code == 404


def test_delete_removes_a_policy(make_admin):
    """A deleted policy stops being enforced and stops being listed."""
    with make_admin() as client:
        client.post(POLICIES, json={"policy": policy_document()})

        response = client.delete(f"{POLICIES}/alcohol_consumption_prohibited")
        registry = client.app.state.policies

    assert response.status_code == 204
    assert len(registry) == 0
    assert registry.restrictions() == []


def test_delete_reports_a_miss(make_admin):
    """Deleting something absent is a 404, not a silent success."""
    with make_admin() as client:
        response = client.delete(f"{POLICIES}/no_such_policy")

    assert response.status_code == 404


# --------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------


def test_no_token_means_no_authentication(make_admin):
    """Without a configured token the routes are open, as documented."""
    with make_admin() as client:
        assert client.get(POLICIES).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}],
    ids=["missing", "wrong-token", "wrong-scheme"],
)
def test_a_configured_token_is_required(make_admin, headers):
    """Anything but the right bearer token is a 401.

    Worth pinning per case: these routes delete guardrails, and the proxy binds every
    interface by default, so a hole here is a hole in the guardrail itself.
    """
    with make_admin(token="s3cret") as client:
        response = client.get(POLICIES, headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_the_right_token_is_accepted(make_admin):
    """The token unlocks every route, not just reads."""
    with make_admin(token="s3cret") as client:
        auth = {"Authorization": "Bearer s3cret"}

        created = client.post(
            POLICIES, json={"policy": policy_document()}, headers=auth
        )
        deleted = client.delete(f"{POLICIES}/11", headers=auth)

    assert created.status_code == 201
    assert deleted.status_code == 204


def test_an_unauthenticated_write_changes_nothing(make_admin):
    """A rejected request must not have mutated the registry on its way to the 401."""
    with make_admin(token="s3cret") as client:
        response = client.post(POLICIES, json={"policy": policy_document()})
        registry = client.app.state.policies

    assert response.status_code == 401
    assert len(registry) == 0
