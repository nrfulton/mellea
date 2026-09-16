# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PolicyProxy and policy loading."""

import tempfile
from pathlib import Path

import pytest

from cli.proxy.policy import PolicyConstraints, PolicyFile, Risk, load_policy

SAMPLE_POLICY_YAML = """\
risk_group: test_policy_group
risk_group_id: 99
description: Test policy for unit tests
policy_version: v1.0
risks:
  - risk: test_risk
    risk_id: 99.1
    description: A test risk
    reason_denial: TEST_DENIED
    short_reply_type: EXPLICIT_REFUSAL
    exception: null
    policy:
      reply_cannot_contain:
        - Forbidden content type A
        - Forbidden content type B
      reply_may_contain:
        - Allowed content type
"""


def test_load_policy_parses_yaml_correctly() -> None:
    """Loading a policy YAML file produces correct dataclass structure."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_POLICY_YAML)
        f.flush()
        policy = load_policy(f.name)

    assert isinstance(policy, PolicyFile)
    assert policy.risk_group == "test_policy_group"
    assert policy.risk_group_id == 99
    assert policy.description == "Test policy for unit tests"
    assert policy.policy_version == "v1.0"

    assert len(policy.risks) == 1
    risk = policy.risks[0]
    assert isinstance(risk, Risk)
    assert risk.risk == "test_risk"
    assert risk.risk_id == "99.1"
    assert risk.reason_denial == "TEST_DENIED"
    assert risk.short_reply_type == "EXPLICIT_REFUSAL"
    assert risk.exception is None

    assert isinstance(risk.policy, PolicyConstraints)
    assert len(risk.policy.reply_cannot_contain) == 2
    assert "Forbidden content type A" in risk.policy.reply_cannot_contain
    assert len(risk.policy.reply_may_contain) == 1


def test_load_policy_file_not_found() -> None:
    """Loading a nonexistent policy file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_policy("/nonexistent/path/policy.yaml")


def test_load_policy_with_path_object() -> None:
    """Loading a policy using a Path object works correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_POLICY_YAML)
        f.flush()
        policy = load_policy(Path(f.name))

    assert policy.risk_group == "test_policy_group"


def test_load_policy_multiple_risks() -> None:
    """Loading a policy with multiple risks parses all of them."""
    multi_risk_yaml = """\
risk_group: multi_risk_group
risk_group_id: 100
description: Policy with multiple risks
policy_version: v1.0
risks:
  - risk: first_risk
    risk_id: 100.1
    description: First risk
    reason_denial: FIRST_DENIED
    short_reply_type: EXPLICIT_REFUSAL
    policy:
      reply_cannot_contain:
        - First forbidden
  - risk: second_risk
    risk_id: 100.2
    description: Second risk
    reason_denial: SECOND_DENIED
    short_reply_type: CAUTIOUS_INFORMATIVE
    policy:
      reply_cannot_contain:
        - Second forbidden
      reply_may_contain:
        - Second allowed
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(multi_risk_yaml)
        f.flush()
        policy = load_policy(f.name)

    assert len(policy.risks) == 2
    assert policy.risks[0].risk_id == "100.1"
    assert policy.risks[1].risk_id == "100.2"
    assert policy.risks[0].short_reply_type == "EXPLICIT_REFUSAL"
    assert policy.risks[1].short_reply_type == "CAUTIOUS_INFORMATIVE"


def test_policy_constraints_defaults() -> None:
    """PolicyConstraints uses empty lists as defaults."""
    constraints = PolicyConstraints()
    assert constraints.reply_cannot_contain == []
    assert constraints.reply_may_contain == []


def test_extract_text_from_chunks() -> None:
    """Extracting text from streaming chunks concatenates all content."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    from cli.proxy.policy_proxy import _extract_text_from_chunks

    chunks = [
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content=" world"), finish_reason=None)
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="!"), finish_reason="stop")
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        ),
    ]

    result = _extract_text_from_chunks(chunks)
    assert result == "Hello world!"


def test_extract_text_from_empty_chunks() -> None:
    """Extracting text from empty chunks returns empty string."""
    from cli.proxy.policy_proxy import _extract_text_from_chunks

    assert _extract_text_from_chunks([]) == ""


def test_create_refusal_chunks() -> None:
    """Creating refusal chunks produces a single chunk with the refusal message."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    from cli.proxy.policy_proxy import _create_refusal_chunks

    original_chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        choices=[
            Choice(index=0, delta=ChoiceDelta(content="test"), finish_reason=None)
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )

    refusal_message = "Policy violation"
    result = _create_refusal_chunks(refusal_message, original_chunk)

    assert len(result) == 1
    assert result[0].id == "chatcmpl-123"
    assert result[0].model == "test-model"
    assert result[0].choices[0].delta.content == "Policy violation"
    assert result[0].choices[0].finish_reason == "stop"


def test_mproxy_requires_streaming_buffer_default_false() -> None:
    """Default MProxy.requires_streaming_buffer is False."""
    from cli.proxy.mproxy import PassthroughProxy

    proxy = PassthroughProxy()
    assert proxy.requires_streaming_buffer is False


def test_policy_proxy_requires_streaming_buffer_true() -> None:
    """PolicyProxy.requires_streaming_buffer is True for policy enforcement."""
    from unittest.mock import patch

    # Mock load_policies to avoid needing a real file
    with patch("cli.proxy.policy_proxy.load_policies", return_value=[]):
        from cli.proxy.policy_proxy import PolicyProxy

        proxy = PolicyProxy(policy_paths=[])
        assert proxy.requires_streaming_buffer is True
