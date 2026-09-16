# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural policies in the `granite.trust.policy-tools` YAML format.

Parses and stores the schema published by
[ibm-granite/granite.trust.policy-tools](https://github.com/ibm-granite/granite.trust.policy-tools),
in which a document describes one risk group and each risk under it lists what a model
reply `reply_cannot_contain` and what it `reply_may_contain`:

```yaml
risk_group: alcohol_consumption_prohibited
risk_group_id: 11
description: Policy for jurisdictions where alcohol is prohibited by law.
policy_version: v1.0
risks:
  - risk: alcohol_general_requests
    risk_id: 11.1
    description: Requests for information about alcohol
    reason_denial: ALCOHOL_PROHIBITED
    short_reply_type: EXPLICIT_REFUSAL
    exception: ALCOHOL_REQUEST_EXCEPTION
    policy:
      reply_cannot_contain:
        - Recommendations for alcoholic beverages
      reply_may_contain:
        - Polite refusal explaining that alcohol-related assistance is unavailable
```

Because the restrictions describe a *reply*, they are enforced by a response hook rather
than a request hook -- see `policy_guard` in `cli.mitm.hooks`. This module holds no
model code, so a policy can be loaded and inspected without any weights present.

A document is read either as YAML text with `load_policy` or as an already-parsed mapping
with `policy_from_mapping`, and written back out as a mapping with `policy_to_mapping`.
The second pair exists for callers that carry a policy in some other encoding -- the
`cli.mitm.admin` control plane accepts and returns JSON -- so that they validate a policy
through exactly the same code, and report exactly the same errors, as the YAML path.

Nothing here interprets `short_reply_type`: upstream documents it as advisory guidance
for whatever enforcement mechanism consumes the policy, and defines no text for its
values. It is parsed and carried so callers can act on it, and ignored by `policy_guard`.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml

__all__ = [
    "Policy",
    "PolicyEntry",
    "PolicyError",
    "PolicyRegistry",
    "PolicyRisk",
    "load_policy",
    "policy_from_mapping",
    "policy_to_mapping",
]

POLICY_SUFFIXES = (".yaml", ".yml")
"""File extensions treated as policy documents when a directory is loaded."""


class PolicyError(ValueError):
    """Raised when a document cannot be read as a policy.

    Covers unparseable YAML and YAML that parses but does not match the schema, so
    callers have one exception type to catch when loading untrusted input.
    """


@dataclass(frozen=True)
class PolicyRisk:
    """One risk entry within a policy.

    Frozen and tuple-valued so that a risk can be read from any thread without
    copying, and so a `PolicyRegistry` snapshot cannot be mutated by its caller.

    Args:
        risk: Machine-readable risk name, e.g. `alcohol_general_requests`.
        risk_id: Dotted identifier as written in the document, e.g. `11.1`. Held as a
            string because YAML parses these as numbers.
        description: What kind of request the risk covers.
        reason_denial: Code explaining why a reply is denied, or `None`.
        short_reply_type: Upstream's advisory hint at how an enforcement mechanism might
            deterministically rewrite the reply, or `None`. Carried, not interpreted.
        exception: Code naming the documented exception to this risk, or `None`.
        reply_cannot_contain: The restrictions a reply violates the policy by containing.
        reply_may_contain: What a compliant reply is permitted to contain, used as
            guidance when composing a replacement for a violating reply.
    """

    risk: str
    risk_id: str
    description: str
    reason_denial: str | None
    short_reply_type: str | None
    exception: str | None
    reply_cannot_contain: tuple[str, ...]
    reply_may_contain: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    """One policy document: a named risk group and the risks it covers.

    Args:
        risk_group: Machine-readable group name, unique within a `PolicyRegistry`.
        risk_group_id: Group identifier as written in the document. Held as a string
            because YAML parses these as numbers.
        description: What the group covers and where it is meant to be deployed.
        policy_version: Schema version string, e.g. `v1.0`.
        risks: The risks in document order.
    """

    risk_group: str
    risk_group_id: str
    description: str
    policy_version: str
    risks: tuple[PolicyRisk, ...]


@dataclass(frozen=True)
class PolicyEntry:
    """A registered policy together with whether it is currently being enforced.

    Enablement is registry state rather than document content. The
    `granite.trust.policy-tools` schema defines no field for it, so it is held beside the
    parsed `Policy` rather than inside it: that keeps a policy a faithful reading of its
    document, and keeps `policy_to_mapping` from writing back a field upstream would not
    recognise.

    Args:
        policy: The parsed policy.
        enabled: Whether `PolicyRegistry.restrictions` includes this policy. A disabled
            policy stays registered, listed, and editable, but is not enforced -- which is
            how a guard is parked without losing its text.
    """

    policy: Policy
    enabled: bool


PolicySnapshot: TypeAlias = list[Policy]
"""A point-in-time copy of a registry's policies."""

PolicyEntrySnapshot: TypeAlias = list[PolicyEntry]
"""A point-in-time copy of a registry's policies with their enablement."""

RestrictionSnapshot: TypeAlias = list[tuple[Policy, PolicyRisk, str]]
"""A point-in-time copy of a registry's `(policy, risk, restriction)` triples."""

# These aliases exist because `PolicyRegistry.list` shadows the `list` builtin inside the
# class body, so annotations written there cannot spell `list[...]` directly.


def _text_list(value: Any, *, field: str) -> tuple[str, ...]:
    """Read a YAML list of restriction strings.

    Empty list entries are dropped rather than rejected: the published
    `policy_schema/schema_v1.0.yaml` template ships bare `-` bullets, which parse as
    `None`, and a policy derived from that template should still load.

    Args:
        value: The parsed YAML value, expected to be a list or `None`.
        field: Field name, used in the error message.

    Returns:
        The non-empty entries, stripped, in document order.

    Raises:
        PolicyError: If `value` is neither a list nor absent.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PolicyError(f"{field!r} must be a list, got {type(value).__name__}.")
    return tuple(
        str(item).strip() for item in value if item is not None and str(item).strip()
    )


def _optional_str(value: Any) -> str | None:
    """Normalise an optional scalar field to a string or `None`.

    Args:
        value: The parsed YAML value. YAML `null` and the empty string both mean absent.

    Returns:
        The value as a string, or `None` if it was absent or empty.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_risk(raw: Any, *, index: int) -> PolicyRisk:
    """Read one entry of a document's `risks` list.

    Args:
        raw: The parsed YAML value for this entry.
        index: Position in the list, used in error messages.

    Returns:
        The parsed risk.

    Raises:
        PolicyError: If the entry is not a mapping, has no `risk` name, or has a
            `policy` block or restriction list of the wrong type.
    """
    if not isinstance(raw, dict):
        raise PolicyError(
            f"risks[{index}] must be a mapping, got {type(raw).__name__}."
        )

    name = _optional_str(raw.get("risk"))
    if name is None:
        raise PolicyError(f"risks[{index}] is missing a 'risk' name.")

    policy = raw.get("policy") or {}
    if not isinstance(policy, dict):
        raise PolicyError(
            f"risks[{index}] 'policy' must be a mapping, got {type(policy).__name__}."
        )

    return PolicyRisk(
        risk=name,
        risk_id=_optional_str(raw.get("risk_id")) or "",
        description=_optional_str(raw.get("description")) or "",
        reason_denial=_optional_str(raw.get("reason_denial")),
        short_reply_type=_optional_str(raw.get("short_reply_type")),
        exception=_optional_str(raw.get("exception")),
        reply_cannot_contain=_text_list(
            policy.get("reply_cannot_contain"),
            field=f"risks[{index}].policy.reply_cannot_contain",
        ),
        reply_may_contain=_text_list(
            policy.get("reply_may_contain"),
            field=f"risks[{index}].policy.reply_may_contain",
        ),
    )


def policy_from_mapping(raw: Any) -> Policy:
    """Read a policy from an already-parsed document.

    The validating half of `load_policy`, split out so that a caller holding a policy in
    some other encoding -- JSON over the `cli.mitm.admin` control plane, say -- gets the
    same checks and the same error messages as the YAML path rather than a second,
    drifting implementation of the schema.

    Args:
        raw: The parsed document, expected to be a mapping.

    Returns:
        The parsed policy.

    Raises:
        PolicyError: If the document is not a mapping, has no `risk_group` name, or has a
            `risks` list the schema does not allow.
    """
    if not isinstance(raw, dict):
        got = "an empty document" if raw is None else type(raw).__name__
        raise PolicyError(f"Policy must be a YAML mapping, got {got}.")

    risk_group = _optional_str(raw.get("risk_group"))
    if risk_group is None:
        raise PolicyError("Policy is missing a 'risk_group' name.")

    risks = raw.get("risks") or []
    if not isinstance(risks, list):
        raise PolicyError(f"'risks' must be a list, got {type(risks).__name__}.")

    return Policy(
        risk_group=risk_group,
        risk_group_id=_optional_str(raw.get("risk_group_id")) or "",
        description=_optional_str(raw.get("description")) or "",
        policy_version=_optional_str(raw.get("policy_version")) or "",
        risks=tuple(_parse_risk(risk, index=i) for i, risk in enumerate(risks)),
    )


def policy_to_mapping(policy: Policy) -> dict[str, Any]:
    """Write a policy back out as a document mapping.

    The inverse of `policy_from_mapping`, and round-trips through it unchanged. Absent
    optional fields are written as `None` rather than omitted, so an editor reading this
    sees every field the schema defines instead of having to know which ones can be
    missing.

    Args:
        policy: The policy to write.

    Returns:
        A mapping in the `granite.trust.policy-tools` schema, ready to serialise as JSON
        or hand to `yaml.safe_dump`.
    """
    return {
        "risk_group": policy.risk_group,
        "risk_group_id": policy.risk_group_id,
        "description": policy.description,
        "policy_version": policy.policy_version,
        "risks": [
            {
                "risk": risk.risk,
                "risk_id": risk.risk_id,
                "description": risk.description,
                "reason_denial": risk.reason_denial,
                "short_reply_type": risk.short_reply_type,
                "exception": risk.exception,
                "policy": {
                    "reply_cannot_contain": list(risk.reply_cannot_contain),
                    "reply_may_contain": list(risk.reply_may_contain),
                },
            }
            for risk in policy.risks
        ],
    }


def load_policy(text: str) -> Policy:
    """Parse a policy document.

    Args:
        text: The YAML document.

    Returns:
        The parsed policy.

    Raises:
        PolicyError: If the YAML is unparseable, is not a mapping, has no `risk_group`
            name, or has a `risks` list the schema does not allow.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PolicyError(f"Policy is not valid YAML: {e}") from e

    return policy_from_mapping(raw)


class PolicyRegistry:
    """The set of policies a proxy is enforcing, mutable while it runs.

    Reachable as `app.state.policies` on an app built by `cli.mitm.app.build_app`, so
    embedding code can load, edit, disable, and drop policies without restarting the
    server; the next request screened by `policy_guard` sees the change. `cli.mitm.admin`
    exposes exactly that over HTTP.

    A policy is either enabled, in which case `restrictions` reports it and it is
    enforced, or disabled, in which case it stays registered and listed but is not. That
    is the difference between parking a guard and deleting it.

    Reads hand out immutable snapshots under a lock, because the hook reads from the
    event loop while the mutating methods may be called from another thread.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._lock = threading.Lock()
        self._entries: dict[str, PolicyEntry] = {}

    def _resolve_locked(self, key: str) -> str | None:
        """Find the `risk_group` a caller's key refers to.

        Callers must already hold `self._lock`; this reads the entry map.

        Args:
            key: A policy's `risk_group` name or its `risk_group_id`.

        Returns:
            The matching `risk_group` name, or `None` if nothing matched. An empty key
            never matches, so it cannot collide with the policies that have no
            `risk_group_id` and therefore carry it as the empty string.
        """
        wanted = str(key).strip()
        if not wanted:
            return None
        if wanted in self._entries:
            return wanted
        for name, entry in self._entries.items():
            if entry.policy.risk_group_id == wanted:
                return name
        return None

    def _register(self, policy: Policy, *, enabled: bool | None) -> Policy:
        """Store a parsed policy, replacing any policy with the same `risk_group`.

        Args:
            policy: The policy to store.
            enabled: Whether to enforce it, or `None` to decide as `add` documents.

        Returns:
            The policy, for the caller to hand back.
        """
        with self._lock:
            existing = self._entries.get(policy.risk_group)
            if enabled is None:
                active = existing.enabled if existing is not None else True
            else:
                active = enabled
            self._entries[policy.risk_group] = PolicyEntry(
                policy=policy, enabled=active
            )
        return policy

    def add(self, text: str, *, enabled: bool | None = None) -> Policy:
        """Parse a policy document and register it.

        Args:
            text: The YAML document.
            enabled: Whether to enforce the policy. The default of `None` preserves the
                flag of the policy being replaced, and enables a policy being registered
                for the first time -- so editing a parked guard does not silently re-arm
                it. Pass a bool to set the flag outright.

        Returns:
            The parsed policy. A policy whose `risk_group` is already registered
            replaces the existing one.

        Raises:
            PolicyError: If the document does not parse as a policy. The registry is
                left unchanged.
        """
        return self._register(load_policy(text), enabled=enabled)

    def add_mapping(self, raw: Any, *, enabled: bool | None = None) -> Policy:
        """Register a policy from an already-parsed document.

        The mapping counterpart of `add`, for callers whose policies did not arrive as
        YAML text.

        Args:
            raw: The parsed document, expected to be a mapping.
            enabled: Whether to enforce the policy, with the same meaning as in `add`.

        Returns:
            The parsed policy, replacing any policy with the same `risk_group`.

        Raises:
            PolicyError: If the document does not parse as a policy. The registry is
                left unchanged.
        """
        return self._register(policy_from_mapping(raw), enabled=enabled)

    def add_path(self, path: str | Path) -> PolicySnapshot:
        """Load policies from a file, or from every policy file in a directory.

        Args:
            path: A YAML file, or a directory whose `.yaml` and `.yml` children are
                each loaded in filename order.

        Returns:
            The policies added, in load order.

        Raises:
            PolicyError: If the path does not exist, a file cannot be read, or a
                document does not parse. The error names the offending file, and
                policies loaded before it stay registered.
        """
        target = Path(path)
        if target.is_dir():
            files = sorted(
                child
                for child in target.iterdir()
                if child.is_file() and child.suffix.lower() in POLICY_SUFFIXES
            )
        elif target.is_file():
            files = [target]
        else:
            raise PolicyError(f"Policy path does not exist: {target}")

        added = []
        for file in files:
            try:
                text = file.read_text(encoding="utf-8")
            except OSError as e:
                raise PolicyError(f"Could not read policy {file}: {e}") from e
            try:
                added.append(self.add(text))
            except PolicyError as e:
                raise PolicyError(f"{file}: {e}") from e
        return added

    def remove(self, key: str) -> bool:
        """Stop enforcing a policy and forget it.

        Args:
            key: The policy's `risk_group` name or its `risk_group_id`.

        Returns:
            `True` if a policy was removed, `False` if no policy matched.
        """
        with self._lock:
            name = self._resolve_locked(key)
            if name is None:
                return False
            del self._entries[name]
            return True

    def set_enabled(self, key: str, enabled: bool) -> bool:
        """Start or stop enforcing a policy without forgetting it.

        Args:
            key: The policy's `risk_group` name or its `risk_group_id`.
            enabled: `True` to include the policy in `restrictions`, `False` to park it.

        Returns:
            `True` if a policy matched, `False` if none did. Setting the flag a policy
            already has succeeds and changes nothing.
        """
        with self._lock:
            name = self._resolve_locked(key)
            if name is None:
                return False
            self._entries[name] = PolicyEntry(
                policy=self._entries[name].policy, enabled=enabled
            )
            return True

    def get(self, key: str) -> PolicyEntry | None:
        """Look up one registered policy.

        Args:
            key: The policy's `risk_group` name or its `risk_group_id`.

        Returns:
            The entry, with its enablement, or `None` if no policy matched.
        """
        with self._lock:
            name = self._resolve_locked(key)
            return self._entries[name] if name is not None else None

    def entries(self) -> PolicyEntrySnapshot:
        """Return the registered policies with their enablement.

        Returns:
            A snapshot in registration order, enabled and disabled alike. Mutating it
            does not affect the registry.
        """
        with self._lock:
            return list(self._entries.values())

    def list(self) -> PolicySnapshot:
        """Return the registered policies.

        Returns:
            A snapshot in registration order, enabled and disabled alike -- this reports
            what is registered, not what is being enforced. Mutating it does not affect
            the registry.
        """
        with self._lock:
            return [entry.policy for entry in self._entries.values()]

    def restrictions(self) -> RestrictionSnapshot:
        """Flatten every restriction across every *enabled* policy.

        This is the order `policy_guard` screens a reply in, so restrictions from
        policies registered earlier are checked first. Disabled policies contribute
        nothing, which is what makes disabling one stop enforcement.

        Returns:
            One `(policy, risk, restriction)` triple per entry in every risk's
            `reply_cannot_contain` list.
        """
        return [
            (entry.policy, risk, restriction)
            for entry in self.entries()
            if entry.enabled
            for risk in entry.policy.risks
            for restriction in risk.reply_cannot_contain
        ]

    def __len__(self) -> int:
        """Return the number of registered policies, enabled and disabled alike."""
        with self._lock:
            return len(self._entries)
