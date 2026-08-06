# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for guardian shim app."""

import pytest

from cli.guardian.app import _build_context_from_messages, _create_decline_response
from cli.serve.models import ChatCompletionRequest, ChatMessage


class TestBuildContextFromMessages:
    """Tests for _build_context_from_messages function."""

    def test_simple_string_messages(self):
        """Test building context from simple string messages."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        context = _build_context_from_messages(messages)

        # Check context has 3 messages using as_list()
        assert len(context.as_list()) == 3

    def test_system_message_included(self):
        """Test that system messages are included in context."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ]
        context = _build_context_from_messages(messages)
        assert len(context.as_list()) == 2

    def test_empty_messages(self):
        """Test building context from empty message list."""
        context = _build_context_from_messages([])
        assert len(context.as_list()) == 0

    def test_none_content_handled(self):
        """Test that None content is handled gracefully."""
        messages = [ChatMessage(role="user", content=None)]
        context = _build_context_from_messages(messages)
        assert len(context.as_list()) == 1


class TestCreateDeclineResponse:
    """Tests for _create_decline_response function."""

    def test_decline_response_structure(self):
        """Test decline response has correct structure."""
        response = _create_decline_response(
            decline_message="Sorry, can't help.",
            model="test-model",
            completion_id="test-123",
            created=1234567890,
        )

        assert response.id == "test-123"
        assert response.model == "test-model"
        assert response.created == 1234567890
        assert response.object == "chat.completion"
        assert len(response.choices) == 1

    def test_decline_response_content(self):
        """Test decline response contains correct content."""
        response = _create_decline_response(
            decline_message="Policy violation.",
            model="test-model",
            completion_id="test-123",
            created=1234567890,
        )

        choice = response.choices[0]
        assert choice.index == 0
        assert choice.finish_reason == "content_filter"
        assert choice.message.content == "Policy violation."
        assert choice.message.role == "assistant"
        assert choice.message.refusal == "Policy violation."

    def test_decline_response_multiline_message(self):
        """Test decline response handles multiline messages."""
        multiline_msg = "Line 1.\nLine 2.\nLine 3."
        response = _create_decline_response(
            decline_message=multiline_msg,
            model="test-model",
            completion_id="test-123",
            created=1234567890,
        )

        assert response.choices[0].message.content == multiline_msg


class TestChatCompletionRequestValidation:
    """Tests for request validation in guardian context."""

    def test_basic_request_parsing(self):
        """Test that basic requests are parsed correctly."""
        request = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}]
        )
        assert request.model == "test-model"
        assert len(request.messages) == 1

    def test_request_with_n_greater_than_one(self):
        """Test that n > 1 is parsed (validation happens in endpoint)."""
        request = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}], n=5
        )
        assert request.n == 5
