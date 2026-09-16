# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PolicyProxy: MProxy implementation that enforces granite.trust.policy-tools policies.

This module provides a proxy that intercepts LLM responses and checks them against
policy constraints defined in YAML files following the granite.trust.policy-tools format.
"""

from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.completion_create_params import CompletionCreateParamsBase

from cli.proxy.mproxy import MProxy
from cli.proxy.policy import PolicyFile, Risk, load_policies


def _extract_text_from_chunks(chunks: list[ChatCompletionChunk]) -> str:
    """Extract the complete text content from a list of streaming chunks.

    Args:
        chunks: List of streaming chunks from the response.

    Returns:
        Concatenated text content from all chunks.
    """
    text_parts: list[str] = []
    for chunk in chunks:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                text_parts.append(delta.content)
    return "".join(text_parts)


def _create_refusal_chunks(
    refusal_message: str, original_chunk: ChatCompletionChunk
) -> list[ChatCompletionChunk]:
    """Create streaming chunks for a refusal message.

    Args:
        refusal_message: The refusal message to stream.
        original_chunk: An original chunk to use as a template for metadata.

    Returns:
        List of chunks that stream the refusal message.
    """
    # Create a single chunk with the full refusal message and finish_reason
    chunk = ChatCompletionChunk(
        id=original_chunk.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=refusal_message),
                finish_reason="stop",
            )
        ],
        created=original_chunk.created,
        model=original_chunk.model,
        object="chat.completion.chunk",
    )
    return [chunk]


def _build_policy_check_prompt(response_text: str, policies: list[PolicyFile]) -> str:
    """Build a prompt for checking response against policies.

    Args:
        response_text: The LLM response text to check.
        policies: List of loaded policy files.

    Returns:
        Formatted prompt for the policy checking LLM.
    """
    constraints_text = []

    for policy in policies:
        constraints_text.append(f"## Policy: {policy.risk_group}")
        constraints_text.append(f"Description: {policy.description}")
        constraints_text.append("")

        for risk in policy.risks:
            if risk.policy.reply_cannot_contain:
                constraints_text.append(f"### Risk: {risk.risk} ({risk.risk_id})")
                constraints_text.append(f"Description: {risk.description}")
                constraints_text.append("The response MUST NOT contain:")
                for item in risk.policy.reply_cannot_contain:
                    constraints_text.append(f"  - {item}")
                constraints_text.append("")

    return f"""You are a policy compliance checker. Analyze the following LLM response and determine if it violates any of the specified policies.

{chr(10).join(constraints_text)}

---

RESPONSE TO CHECK:
{response_text}

---

Analyze the response against ALL policy constraints above. For each constraint violated, identify:
1. The risk_id of the violated policy
2. Which specific "reply_cannot_contain" rule was violated
3. A brief explanation

Respond in exactly this format:
VERDICT: PASS or FAIL
VIOLATIONS: (if FAIL, list each violation as "risk_id: explanation", one per line; if PASS, write "None")
"""


def _parse_verdict(checker_response: str) -> tuple[bool, list[tuple[str, str]]]:
    """Parse the verdict from the policy checker response.

    Args:
        checker_response: Raw response from the policy checking LLM.

    Returns:
        Tuple of (passed, violations) where violations is list of (risk_id, explanation).
    """
    lines = checker_response.strip().split("\n")
    passed = True
    violations: list[tuple[str, str]] = []

    for line in lines:
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict_text = line.split(":", 1)[1].strip().upper()
            passed = verdict_text == "PASS"
        elif ":" in line and not line.upper().startswith("VIOLATIONS:"):
            # Try to parse as "risk_id: explanation"
            if not passed:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    risk_id = parts[0].strip()
                    explanation = parts[1].strip()
                    if risk_id and not risk_id.upper().startswith("VERDICT"):
                        violations.append((risk_id, explanation))

    return passed, violations


def _find_risk_by_id(policies: list[PolicyFile], risk_id: str) -> Risk | None:
    """Find a risk definition by its ID across all policies.

    Args:
        policies: List of policy files to search.
        risk_id: The risk ID to find.

    Returns:
        The Risk if found, None otherwise.
    """
    for policy in policies:
        for risk in policy.risks:
            if risk.risk_id == risk_id:
                return risk
    return None


def _build_refusal_message(
    violations: list[tuple[str, str]], policies: list[PolicyFile]
) -> str:
    """Build a refusal message based on policy violations.

    Args:
        violations: List of (risk_id, explanation) tuples.
        policies: List of policy files for looking up risk details.

    Returns:
        Formatted refusal message.
    """
    if not violations:
        return "I'm sorry, but I cannot provide that response as it violates content policies."

    # Get the first violation's risk for the refusal type
    risk_id, _ = violations[0]
    risk = _find_risk_by_id(policies, risk_id)

    if risk and risk.reason_denial:
        reason = risk.reason_denial.replace("_", " ").title()
        return (
            f"I'm sorry, but I cannot provide that response. "
            f"Reason: {reason}. "
            f"This content is restricted by policy."
        )

    return (
        "I'm sorry, but I cannot provide that response as it violates content policies."
    )


class PolicyProxy(MProxy):
    """MProxy that enforces policies from granite.trust.policy-tools YAML files.

    This proxy checks LLM responses against policy constraints and replaces
    violating responses with appropriate refusal messages.

    The proxy uses a separate LLM call to evaluate whether responses comply
    with the loaded policies. If a response violates any policy constraint,
    it is replaced with a refusal message.

    Example:
        ```python
        from cli.proxy.policy_proxy import PolicyProxy

        proxy = PolicyProxy(
            policy_paths=["policies/alcohol_prohibited.yaml"],
            checker_base_url="http://localhost:11434/v1",
            checker_model="llama3.2",
        )
        ```

    Attributes:
        policies: List of loaded PolicyFile instances.
        checker_client: OpenAI client for policy checking calls.
        checker_model: Model ID to use for policy checking.
    """

    def __init__(
        self,
        policy_paths: list[str | Path],
        checker_base_url: str = "http://localhost:11434/v1",
        checker_model: str = "llama3.2",
        checker_api_key: str = "not-needed",
    ) -> None:
        """Initialize the PolicyProxy.

        Args:
            policy_paths: List of paths to policy YAML files.
            checker_base_url: Base URL for the policy-checking LLM endpoint.
            checker_model: Model ID to use for policy checking.
            checker_api_key: API key for the checker endpoint (if required).
        """
        self.policies = load_policies(policy_paths)
        self.checker_client = OpenAI(base_url=checker_base_url, api_key=checker_api_key)
        self.checker_model = checker_model

    def request_rewrite(
        self, request: CompletionCreateParamsBase
    ) -> CompletionCreateParamsBase:
        """Pass through request unchanged."""
        return request

    def _check_response(self, response_text: str) -> tuple[bool, str]:
        """Check a response against loaded policies.

        Args:
            response_text: The response text to check.

        Returns:
            Tuple of (passed, message) where message is either the original
            response if passed, or a refusal message if failed.
        """
        if not self.policies:
            return True, response_text

        prompt = _build_policy_check_prompt(response_text, self.policies)

        try:
            checker_response = self.checker_client.chat.completions.create(
                model=self.checker_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            checker_text = checker_response.choices[0].message.content or ""
            passed, violations = _parse_verdict(checker_text)

            if passed:
                return True, response_text
            else:
                refusal = _build_refusal_message(violations, self.policies)
                return False, refusal

        except Exception:
            # If policy checking fails, pass through (fail open)
            return True, response_text

    def response_rewrite(self, response: ChatCompletion) -> ChatCompletion:
        """Check response against policies and replace if violating.

        Args:
            response: The original chat completion response.

        Returns:
            The original response if compliant, or a modified response with
            a refusal message if policy violations were detected.
        """
        if not response.choices:
            return response

        original_text = response.choices[0].message.content or ""
        passed, message = self._check_response(original_text)

        if passed:
            return response

        # Create a new response with the refusal message
        new_message = ChatCompletionMessage(role="assistant", content=message)
        new_choice = Choice(index=0, message=new_message, finish_reason="stop")

        return ChatCompletion(
            id=response.id,
            choices=[new_choice],
            created=response.created,
            model=response.model,
            object="chat.completion",
            usage=response.usage,
        )

    def chunk_rewrite(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
        """Pass through streaming chunks unchanged.

        Note: When requires_streaming_buffer is True (the default for PolicyProxy),
        this method is not used. Instead, streaming_rewrite() handles the complete
        buffered response.
        """
        return chunk

    @property
    def requires_streaming_buffer(self) -> bool:
        """PolicyProxy requires buffering to enforce policies on streaming responses."""
        return True

    def streaming_rewrite(
        self, chunks: list[ChatCompletionChunk]
    ) -> list[ChatCompletionChunk] | ChatCompletion:
        """Check buffered streaming response against policies.

        Args:
            chunks: All chunks received from the upstream streaming response.

        Returns:
            The original chunks if compliant, or refusal chunks/response if
            policy violations were detected.
        """
        if not chunks:
            return chunks

        # Extract full text from chunks
        full_text = _extract_text_from_chunks(chunks)

        # Check against policies
        passed, message = self._check_response(full_text)

        if passed:
            return chunks

        # Policy violation - return refusal chunks
        return _create_refusal_chunks(message, chunks[0])
