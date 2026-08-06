# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy model for guardian shim YAML configuration."""

from pathlib import Path

from pydantic import BaseModel, Field

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "The 'm guardian' command requires extra dependencies. "
        'Please install them with: pip install "mellea[server]"'
    ) from e


class PolicyRule(BaseModel):
    """A single policy rule with optional custom decline message."""

    name: str
    """Short name for the policy rule (used in logs)."""

    policy: str
    """The policy text that the guardian adapter will check against."""

    decline_message: str | None = None
    """Custom message to return when this policy is violated. If None, uses default."""


class GuardianCriteria(BaseModel):
    """Guardian-core criteria check configuration."""

    name: str
    """Name of the criteria (can be a key from CRITERIA_BANK or custom text)."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Risk score threshold above which to decline (0.0-1.0)."""

    scoring_schema: str = "user_prompt"
    """Which part of the conversation to evaluate (user_prompt, assistant_response, etc)."""

    decline_message: str | None = None
    """Custom message to return when this criteria is triggered."""


class GuardianPolicy(BaseModel):
    """Root policy configuration for the guardian shim."""

    version: str = "1.0"
    """Schema version for forward compatibility."""

    default_decline_message: str = (
        "I'm sorry, but I can't help with that request as it doesn't "
        "align with our usage policies."
    )
    """Default message when a request is declined."""

    policy_rules: list[PolicyRule] = Field(default_factory=list)
    """List of policy-guardrails checks to run on each request."""

    guardian_criteria: list[GuardianCriteria] = Field(default_factory=list)
    """List of guardian-core criteria checks to run on each request."""


def load_policy(policy_path: str | Path) -> GuardianPolicy:
    """Load a guardian policy from a YAML file.

    Args:
        policy_path: Path to the YAML policy file.

    Returns:
        Parsed GuardianPolicy object.

    Raises:
        FileNotFoundError: If the policy file doesn't exist.
        ValueError: If the YAML is invalid or doesn't match the schema.
    """
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return GuardianPolicy.model_validate(data)
