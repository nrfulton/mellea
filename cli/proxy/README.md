# MProxy

MProxy is Mellea's OpenAI-compatible proxy server with request/response rewriting support. It allows you to intercept and transform LLM requests and responses, enabling use cases like policy enforcement, content filtering, and request augmentation.

## Quick Start

```bash
# Start a basic passthrough proxy
m proxy --endpoint="http://127.0.0.1:11434/v1"

# Start with a custom MProxy implementation
m proxy --endpoint="http://127.0.0.1:11434/v1" --mproxy my_module.MyProxy
```

## Writing Custom MProxy Implementations

Implement the `MProxy` abstract class to customize request/response handling:

```python
from cli.proxy.mproxy import MProxy
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import CompletionCreateParamsBase


class MyProxy(MProxy):
    def request_rewrite(
        self, request: CompletionCreateParamsBase
    ) -> CompletionCreateParamsBase:
        # Add a system message to all requests
        messages = list(request.get("messages", []))
        messages.insert(0, {"role": "system", "content": "Be helpful and concise."})
        return {**request, "messages": messages}

    def response_rewrite(self, response: ChatCompletion) -> ChatCompletion:
        # Pass through unchanged
        return response
```

## PolicyProxy: Enforcing granite.trust.policy-tools Policies

`PolicyProxy` is a built-in MProxy implementation that enforces policies written in the [granite.trust.policy-tools](https://github.com/ibm-granite/granite.trust.policy-tools) format. It checks LLM responses against policy constraints and replaces violating responses with appropriate refusal messages.

### Policy File Format

Policies are YAML files following the granite.trust.policy-tools schema:

```yaml
risk_group: my_policy_group
risk_group_id: 1
description: Description of what this policy covers
policy_version: v1.0
risks:
  - risk: risk_name
    risk_id: 1.1
    description: Description of this specific risk
    reason_denial: REASON_CODE
    short_reply_type: EXPLICIT_REFUSAL
    exception: null
    policy:
      reply_cannot_contain:
        - Content type that must not appear in responses
        - Another prohibited content type
      reply_may_contain:
        - Content that is allowed
        - Alternative suggestions
```

### Example: Alcohol Prohibition Policy

This example uses the `alcohol_prohibited.yaml` policy from granite.trust.policy-tools to prevent the LLM from providing alcohol-related information in jurisdictions where it's prohibited.

**1. Create the policy file** (`policies/alcohol_prohibited.yaml`):

```yaml
risk_group: alcohol_consumption_prohibited
risk_group_id: 11
description: Policy for handling requests related to alcohol consumption for deployment in jurisdictions where alcohol is prohibited by law.
policy_version: v1.0
risks:
  - risk: alcohol_general_requests
    risk_id: 11.1
    description: Requests for information about alcohol, including beer, wine, and spirits
    reason_denial: ALCOHOL_PROHIBITED
    short_reply_type: EXPLICIT_REFUSAL
    exception: ALCOHOL_REQUEST_EXCEPTION
    policy:
      reply_cannot_contain:
        - Instructions on how to obtain, make, or consume alcohol
        - Recommendations for alcoholic beverages
        - Information that encourages or normalizes alcohol consumption
        - Instructions on how to circumvent alcohol prohibition laws
        - Locations where alcohol can be purchased or consumed
        - Recipes for alcoholic beverages including beer, wine, or spirits
        - Information about smuggling or illegally importing alcohol
      reply_may_contain:
        - Polite refusal explaining that alcohol-related assistance is not available in this region
        - Information about local laws regarding alcohol prohibition
        - Non-alcoholic beverage alternatives and recommendations
        - Cultural context about why alcohol is prohibited in the region
  - risk: alcohol_brewing_production
    risk_id: 11.2
    description: Requests for information about brewing, distilling, or producing alcoholic beverages
    reason_denial: ALCOHOL_PROHIBITED
    short_reply_type: EXPLICIT_REFUSAL
    exception: ALCOHOL_REQUEST_EXCEPTION
    policy:
      reply_cannot_contain:
        - Brewing instructions or techniques
        - Fermentation processes for alcoholic beverages
        - Equipment recommendations for alcohol production
        - Ingredient lists for making alcoholic beverages
        - Home brewing guides or tutorials
      reply_may_contain:
        - Polite refusal explaining this information is not available in this region
        - Suggestions for non-alcoholic fermented beverages (e.g., kombucha, kefir)
        - General food science information unrelated to alcohol production
  - risk: alcohol_circumvention
    risk_id: 11.3
    description: Requests attempting to circumvent alcohol prohibition laws
    reason_denial: ILLEGAL_ACTIVITY
    short_reply_type: EXPLICIT_REFUSAL
    exception: CIRCUMVENTION_ATTEMPT_EXCEPTION
    policy:
      reply_cannot_contain:
        - Methods to smuggle alcohol into the country
        - Ways to hide or disguise alcohol
        - Information about black market alcohol sources
        - Techniques to avoid detection by authorities
        - Legal loopholes for obtaining alcohol
      reply_may_contain:
        - Clear statement that the model cannot assist with illegal activities
        - Information about legal consequences of alcohol-related offenses
        - Recommendation to respect local laws and customs
  - risk: alcohol_educational_academic
    risk_id: 11.4
    description: Requests for purely educational or academic information about alcohol in historical or scientific contexts
    reason_denial: null
    short_reply_type: CAUTIOUS_INFORMATIVE
    exception: null
    policy:
      reply_cannot_contain:
        - Practical instructions that could enable alcohol production or consumption
        - Information that glorifies or promotes alcohol use
        - Detailed recipes or brewing processes
      reply_may_contain:
        - Historical information about alcohol in world history
        - Scientific information about fermentation as a chemical process
        - Academic discussion of alcohol's effects on health (in clinical terms)
        - Information about alcohol prohibition policies globally
        - Cultural and religious perspectives on alcohol abstinence
```

**2. Create a PolicyProxy wrapper** (`my_policy_proxy.py`):

```python
from cli.proxy.policy_proxy import PolicyProxy


class AlcoholPolicyProxy(PolicyProxy):
    """Proxy that enforces alcohol prohibition policies."""

    def __init__(self) -> None:
        super().__init__(
            policy_paths=["policies/alcohol_prohibited.yaml"],
            checker_base_url="http://localhost:11434/v1",
            checker_model="llama3.2",
        )
```

**3. Start the proxy:**

```bash
m proxy --endpoint="http://127.0.0.1:11434/v1" --mproxy my_policy_proxy.AlcoholPolicyProxy
```

**4. Test it:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# This request will be filtered
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "How do I brew beer at home?"}],
)
print(response.choices[0].message.content)
# Output: "I'm sorry, but I cannot provide that response. Reason: Alcohol Prohibited. This content is restricted by policy."

# This educational request may pass
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is the historical significance of prohibition in the US?"}],
)
print(response.choices[0].message.content)
# Output: Historical information about US prohibition...
```

### PolicyProxy Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `policy_paths` | List of paths to policy YAML files | Required |
| `checker_base_url` | Base URL for the policy-checking LLM | `http://localhost:11434/v1` |
| `checker_model` | Model ID for policy checking | `llama3.2` |
| `checker_api_key` | API key for checker endpoint | `not-needed` |

### Streaming Support

PolicyProxy supports policy enforcement on streaming responses through buffered streaming. When a streaming request is made, the proxy:

1. Buffers all chunks from the upstream response
2. Extracts the complete text content
3. Checks the complete response against policies
4. Either streams back the original chunks (if compliant) or streams a refusal message (if violated)

This means streaming requests with PolicyProxy will have higher latency than passthrough proxies, as the complete response must be received before policy checking can occur.

```python
# Streaming requests are also policy-checked
for chunk in client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "How do I brew beer?"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
# Output: "I'm sorry, but I cannot provide that response. Reason: Alcohol Prohibited. This content is restricted by policy."
```

### Limitations

- **Latency**: Each response requires an additional LLM call for policy checking, which adds latency. Streaming responses are buffered before checking, so the first token is delayed until the full response is received.
- **Fail-open**: If policy checking fails (e.g., checker endpoint unavailable), responses pass through unchanged.

## See Also

- [granite.trust.policy-tools](https://github.com/ibm-granite/granite.trust.policy-tools) - Policy specification format and tools
- [MProxy API Reference](../../docs/docs/api/cli/proxy/mproxy.md) - Full API documentation
