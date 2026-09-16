# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy loading and data structures for granite.trust.policy-tools format.

This module provides dataclasses and utilities for loading policies written in the
style of https://github.com/ibm-granite/granite.trust.policy-tools.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PolicyConstraints:
    """Constraints defining what a reply can and cannot contain.

    Attributes:
        reply_cannot_contain: List of content types/topics that must not appear in replies.
        reply_may_contain: List of content types/topics that are permitted in replies.
    """

    reply_cannot_contain: list[str] = field(default_factory=list)
    reply_may_contain: list[str] = field(default_factory=list)


@dataclass
class Risk:
    """A single risk definition within a policy.

    Attributes:
        risk: Short name identifying the risk.
        risk_id: Unique identifier for this risk (e.g., "11.1").
        description: Human-readable description of the risk.
        reason_denial: Reason code for denying requests (e.g., "ALCOHOL_PROHIBITED").
        short_reply_type: Type of reply when policy is violated (e.g., "EXPLICIT_REFUSAL").
        exception: Exception identifier if applicable.
        policy: Constraints defining allowed and prohibited content.
    """

    risk: str
    risk_id: str
    description: str
    reason_denial: str | None
    short_reply_type: str
    exception: str | None
    policy: PolicyConstraints


@dataclass
class PolicyFile:
    """A complete policy file following the granite.trust.policy-tools format.

    Attributes:
        risk_group: Category name for this policy group.
        risk_group_id: Unique identifier for the risk group.
        description: Human-readable description of the policy.
        policy_version: Version string (e.g., "v1.0").
        risks: List of individual risk definitions.
    """

    risk_group: str
    risk_group_id: int
    description: str
    policy_version: str
    risks: list[Risk]


def load_policy(path: str | Path) -> PolicyFile:
    """Load a policy file from a YAML file.

    Args:
        path: Path to the YAML policy file.

    Returns:
        Parsed PolicyFile instance.

    Raises:
        FileNotFoundError: If the policy file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    risks = []
    for risk_data in data.get("risks", []):
        policy_data = risk_data.get("policy", {})
        constraints = PolicyConstraints(
            reply_cannot_contain=policy_data.get("reply_cannot_contain", []),
            reply_may_contain=policy_data.get("reply_may_contain", []),
        )
        risks.append(
            Risk(
                risk=risk_data["risk"],
                risk_id=str(risk_data["risk_id"]),
                description=risk_data.get("description", ""),
                reason_denial=risk_data.get("reason_denial"),
                short_reply_type=risk_data.get("short_reply_type", "EXPLICIT_REFUSAL"),
                exception=risk_data.get("exception"),
                policy=constraints,
            )
        )

    return PolicyFile(
        risk_group=data["risk_group"],
        risk_group_id=data["risk_group_id"],
        description=data.get("description", ""),
        policy_version=data.get("policy_version", "v1.0"),
        risks=risks,
    )


def load_policies(paths: list[str | Path]) -> list[PolicyFile]:
    """Load multiple policy files.

    Args:
        paths: List of paths to YAML policy files.

    Returns:
        List of parsed PolicyFile instances.
    """
    return [load_policy(p) for p in paths]
