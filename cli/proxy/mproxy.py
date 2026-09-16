# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MProxy interface for request/response rewriting in the proxy server."""

from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsBase


class MProxy(ABC):
    """Interface for request/response rewriting in the proxy server.

    Implement this class to customize how requests are transformed before being
    sent to the upstream endpoint, and how responses are transformed before
    being returned to the client.

    Example:
        ```python
        from cli.proxy.mproxy import MProxy
        from openai.types.chat import ChatCompletion
        from openai.types.chat.completion_create_params import (
            CompletionCreateParamsBase,
        )

        class MyProxy(MProxy):
            def request_rewrite(
                self, request: CompletionCreateParamsBase
            ) -> CompletionCreateParamsBase:
                # Add a system message to all requests
                messages = list(request.get("messages", []))
                messages.insert(0, {"role": "system", "content": "Be helpful."})
                return {**request, "messages": messages}

            def response_rewrite(self, response: ChatCompletion) -> ChatCompletion:
                # Pass through unchanged
                return response
        ```
    """

    @abstractmethod
    def request_rewrite(
        self, request: CompletionCreateParamsBase
    ) -> CompletionCreateParamsBase:
        """Transform a request before sending to the upstream endpoint.

        Args:
            request: The original chat completion request parameters.

        Returns:
            The transformed request parameters to send upstream.
        """
        ...

    @abstractmethod
    def response_rewrite(self, response: ChatCompletion) -> ChatCompletion:
        """Transform a non-streaming response before returning to the client.

        Args:
            response: The original chat completion response from upstream.

        Returns:
            The transformed response to return to the client.
        """
        ...

    def chunk_rewrite(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
        """Transform a streaming chunk before returning to the client.

        Override this method to customize streaming response transformation.
        By default, chunks are passed through unchanged.

        Args:
            chunk: The original streaming chunk from upstream.

        Returns:
            The transformed chunk to return to the client.
        """
        return chunk


class PassthroughProxy(MProxy):
    """Default proxy implementation that passes requests/responses unchanged."""

    def request_rewrite(
        self, request: CompletionCreateParamsBase
    ) -> CompletionCreateParamsBase:
        """Pass through request unchanged."""
        return request

    def response_rewrite(self, response: ChatCompletion) -> ChatCompletion:
        """Pass through response unchanged."""
        return response
