# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for guardian policy models."""

import tempfile
from pathlib import Path

import pytest

from cli.guardian.policy import (
    GuardianCriteria,
    GuardianPolicy,
    PolicyRule,
    load_policy,
)


class TestPolicyRule:
    """Tests for PolicyRule model."""

    def test_basic_rule(self):
        """Test basic policy rule creation."""
        rule = PolicyRule(name="test_rule", policy="No harmful content allowed")
        assert rule.name == "test_rule"
        assert rule.policy == "No harmful content allowed"
        assert rule.decline_message is None

    def test_rule_with_custom_decline(self):
        """Test policy rule with custom decline message."""
        rule = PolicyRule(
            name="test_rule",
            policy="Be polite",
            decline_message="Please be more polite.",
        )
        assert rule.decline_message == "Please be more polite."


class TestGuardianCriteria:
    """Tests for GuardianCriteria model."""

    def test_basic_criteria(self):
        """Test basic criteria creation with defaults."""
        criteria = GuardianCriteria(name="harm")
        assert criteria.name == "harm"
        assert criteria.threshold == 0.5
        assert criteria.scoring_schema == "user_prompt"
        assert criteria.decline_message is None

    def test_criteria_with_custom_values(self):
        """Test criteria with all custom values."""
        criteria = GuardianCriteria(
            name="jailbreak",
            threshold=0.8,
            scoring_schema="assistant_response",
            decline_message="Nice try!",
        )
        assert criteria.name == "jailbreak"
        assert criteria.threshold == 0.8
        assert criteria.scoring_schema == "assistant_response"
        assert criteria.decline_message == "Nice try!"

    def test_threshold_bounds(self):
        """Test threshold validation bounds."""
        # Valid thresholds
        GuardianCriteria(name="test", threshold=0.0)
        GuardianCriteria(name="test", threshold=1.0)
        GuardianCriteria(name="test", threshold=0.5)

        # Invalid thresholds
        with pytest.raises(ValueError):
            GuardianCriteria(name="test", threshold=-0.1)
        with pytest.raises(ValueError):
            GuardianCriteria(name="test", threshold=1.1)


class TestGuardianPolicy:
    """Tests for GuardianPolicy model."""

    def test_default_policy(self):
        """Test empty policy with defaults."""
        policy = GuardianPolicy()
        assert policy.version == "1.0"
        assert "can't help" in policy.default_decline_message.lower()
        assert policy.policy_rules == []
        assert policy.guardian_criteria == []

    def test_policy_with_rules_and_criteria(self):
        """Test policy with both rules and criteria."""
        policy = GuardianPolicy(
            version="1.0",
            default_decline_message="No.",
            policy_rules=[PolicyRule(name="rule1", policy="Be nice")],
            guardian_criteria=[GuardianCriteria(name="harm", threshold=0.7)],
        )
        assert len(policy.policy_rules) == 1
        assert len(policy.guardian_criteria) == 1
        assert policy.policy_rules[0].name == "rule1"
        assert policy.guardian_criteria[0].name == "harm"


class TestLoadPolicy:
    """Tests for load_policy function."""

    def test_load_minimal_policy(self):
        """Test loading a minimal YAML policy."""
        yaml_content = """
version: "1.0"
default_decline_message: "Nope."
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = load_policy(f.name)

        assert policy.version == "1.0"
        assert policy.default_decline_message == "Nope."
        assert policy.policy_rules == []
        assert policy.guardian_criteria == []

    def test_load_full_policy(self):
        """Test loading a full YAML policy with rules and criteria."""
        yaml_content = """
version: "1.0"
default_decline_message: "Default decline."

policy_rules:
  - name: no_harm
    policy: "No harmful content"
    decline_message: "That's harmful."

guardian_criteria:
  - name: harm
    threshold: 0.7
    scoring_schema: user_prompt
  - name: jailbreak
    threshold: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = load_policy(f.name)

        assert len(policy.policy_rules) == 1
        assert policy.policy_rules[0].name == "no_harm"
        assert policy.policy_rules[0].decline_message == "That's harmful."

        assert len(policy.guardian_criteria) == 2
        assert policy.guardian_criteria[0].name == "harm"
        assert policy.guardian_criteria[0].threshold == 0.7
        assert policy.guardian_criteria[1].name == "jailbreak"
        assert policy.guardian_criteria[1].threshold == 0.8

    def test_load_empty_file(self):
        """Test loading an empty YAML file uses defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            policy = load_policy(f.name)

        # Should use all defaults
        assert policy.version == "1.0"
        assert policy.policy_rules == []
        assert policy.guardian_criteria == []

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_policy("/nonexistent/path/policy.yaml")

    def test_load_policy_from_path_object(self):
        """Test loading policy using Path object."""
        yaml_content = "version: '2.0'"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = load_policy(Path(f.name))

        assert policy.version == "2.0"
