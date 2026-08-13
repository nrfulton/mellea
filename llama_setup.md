# Running `ibm-granite/granite-4.1-30b` on llama.cpp and querying it from Mellea

End-to-end log of building llama.cpp with CUDA, serving Granite 4.1 30B via
`llama-server`, and driving it from a Mellea program through Mellea's
OpenAI-compatible backend.

Everything below was executed and verified on this machine on 2026-08-11.

## 1. Baseline

| Property | Value |
| --- | --- |
| GPU | NVIDIA GB10 (DGX Spark), compute capability **12.1** |
| Memory | 121 GB unified (CPU/GPU shared), 124546 MiB visible to CUDA |
| Arch / OS | `aarch64`, Ubuntu 24.04, Linux 6.17 |
| CUDA toolkit | 13.0.88 (`/usr/local/cuda`), driver 580.173.02 |
| Host compiler | GCC 13.3.0 |
| CMake | 3.28.3 |
| Cores | 20 |

Unified memory is the important detail: on a GB10 there is no separate VRAM
pool to fit the model into, so a 30B model at 4-bit is comfortable and even
`bf16` (~58 GB) would load. `nvidia-smi` reports `[N/A]` for `memory.total`
and `memory.used` on this platform — use `llama-cli --list-devices` instead.

## 2. Build llama.cpp with CUDA

```bash
cd ~
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j "$(nproc)"
```

Notes on the flags:

- **`-DCMAKE_CUDA_ARCHITECTURES=121`** is the one arch-specific choice. It must
  match the GPU's compute capability (`nvidia-smi --query-gpu=compute_cap
  --format=csv`). llama.cpp rewrites it to `121a` (the architecture-specific
  Blackwell variant) during configure — that log line is expected:

  ```text
  -- Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a
  -- Using CMAKE_CUDA_ARCHITECTURES=121a CMAKE_CUDA_ARCHITECTURES_NATIVE=121a-real
  ```

- **`-DLLAMA_BUILD_EXAMPLES=OFF`** skips the examples but still builds
  `llama-server`, `llama-cli`, and `llama-bench` (they live under `tools/`, which
  is governed by `LLAMA_BUILD_TOOLS`, not `LLAMA_BUILD_EXAMPLES`).

Two configure warnings are harmless here:

- `Could NOT find NCCL` — only matters for multi-GPU.
- `OpenSSL not found, HTTPS support disabled` — the server still speaks plain
  HTTP, which is all a localhost setup needs.

The build compiled 141 CUDA translation units and finished in **2 min 45 s** at
`-j 20` (fast because only one GPU architecture is targeted). Verify:

```bash
$ ./build/bin/llama-server --version
version: 1 (f785fc9)
built with GNU 13.3.0 for Linux aarch64

$ ./build/bin/llama-cli --list-devices
Available devices:
  CUDA0: NVIDIA GB10 (124546 MiB, 119157 MiB free)
```

If `--list-devices` shows no CUDA device, the CUDA backend did not compile in —
re-check `nvcc --version` and the configure output for `Including CUDA backend`.

## 3. Get the model

IBM publishes official GGUF conversions at
[`ibm-granite/granite-4.1-30b-GGUF`](https://huggingface.co/ibm-granite/granite-4.1-30b-GGUF),
so no manual `convert_hf_to_gguf.py` step is needed.

`granite-4.1-30b` is a **dense** 30B model (`GraniteForCausalLM`, 64 layers,
`hidden_size` 4096, 128K context) — not a mixture-of-experts. That matters for
throughput: every token reads the whole weight tensor set, so decode speed is
bounded by memory bandwidth × quantization size.

| Quant | Size | Notes |
| --- | --- | --- |
| `Q4_K_M` | 17.5 GB | **Used here.** Best speed/quality balance |
| `Q5_K_M` | 20.5 GB | Slightly better quality, ~15% slower |
| `Q6_K` | 23.7 GB | Near-lossless |
| `Q8_0` | 30.7 GB | Effectively lossless; ~1.75× the bytes per token, so decode is proportionally slower |
| `bf16` | 57.7 GB | 5 shards; fits in unified memory but bandwidth-starved |

```bash
mkdir -p ~/models
uv run hf download ibm-granite/granite-4.1-30b-GGUF \
  granite-4.1-30b-Q4_K_M.gguf \
  --local-dir ~/models/granite-4.1-30b-GGUF
```

`hf` ships with the `huggingface_hub` already present in Mellea's venv, hence
`uv run`. The download sustained ~10 MB/s here, about 30 minutes for 17.5 GB.

## 4. Start `llama-server`

```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/models/granite-4.1-30b-GGUF/granite-4.1-30b-Q4_K_M.gguf \
  --alias ibm-granite/granite-4.1-30b \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --jinja
```

| Flag | Why |
| --- | --- |
| `--alias ibm-granite/granite-4.1-30b` | Name reported by `/v1/models` and accepted in the `model` request field. Without it the alias is the file path, which makes client-side model names awkward. |
| `--n-gpu-layers 999` | Offload every layer. Anything ≥ 64 works for this model. |
| `--ctx-size 8192` | Total KV cache across slots. The model supports 131072; raise it if you need long context, at a proportional memory cost. |
| `--jinja` | Use the chat template embedded in the GGUF. **Keep this on** — Granite's template handles the system-prompt and tool-call structure that Mellea's requests rely on. |
| `--host 127.0.0.1` | Loopback only — the safe default, since the server has no auth (see below). Use `--host 0.0.0.0` to accept connections from other machines; see [LAN binding](#lan-binding). |

Startup is fast (~2 s) because weights are memory-mapped:

```text
srv    load_model: loading model '.../granite-4.1-30b-Q4_K_M.gguf'
srv    load_model: initializing, n_slots = 4, n_ctx_slot = 8192, kv_unified = 'true'
srv  llama_server: model loaded
srv  llama_server: listening on http://127.0.0.1:8080
```

Two log messages worth knowing about:

- `CORS is set to allow all origins ('*') and no API key is set` — fine for a
  loopback-bound server, but it is a real exposure once the bind address is
  routable. See [LAN binding](#lan-binding).
- `NOTICE: server default port will be changed to :9931 in a future release` —
  this setup passes `--port 8080` explicitly, so it is unaffected.

### LAN binding

The server is currently running with `--host 0.0.0.0`, i.e. reachable from other
machines on the network:

```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/models/granite-4.1-30b-GGUF/granite-4.1-30b-Q4_K_M.gguf \
  --alias ibm-granite/granite-4.1-30b \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 999 --ctx-size 8192 --jinja
```

Confirmed working from both `127.0.0.1:8080` and this host's LAN address
`9.105.22.23:8080` (`enP7s7`).

Because `llama-server` has **no authentication and permissive CORS by default**,
a `0.0.0.0` bind means anyone who can route to this host can submit prompts and
consume the GPU. If this network is not fully trusted, either:

- add `--api-key <secret>` to the server command and set the same value as the
  Mellea backend's `api_key`, or
- restrict access at the firewall, e.g.
  `sudo ufw allow from <trusted-subnet> to any port 8080`, or
- bind to a specific interface instead of all of them (`--host 9.105.22.23`).

To point the Mellea client at the server from another machine, change one line:

```python
base_url = "http://9.105.22.23:8080/v1"
```

### Verify

```bash
$ curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
{
    "models": [
        {
            "name": "ibm-granite/granite-4.1-30b",
            ...
```

Measured performance (native `/completion` endpoint, which reports `timings`):

| Phase | Throughput |
| --- | --- |
| Prompt eval (prefill, 802 tokens) | **821 tok/s** |
| Generation (decode) | **12.3 tok/s** |

The 821 tok/s prefill is the reliable signal that offload actually happened;
CPU-only prefill on a 30B dense model is an order of magnitude slower (this was
not measured here — the comparison is offered as a sanity check, not a
benchmark). A useful one-liner for re-checking:

```bash
curl -s http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write one paragraph about tokenizers.","n_predict":128,"temperature":0}' \
| python3 -c "
import json,sys
t = json.load(sys.stdin)['timings']
print(f\"gen: {t['predicted_n']} tok @ {t['predicted_per_second']:.1f} tok/s\")
"
```

## 5. The Mellea program

`llama-server` exposes an OpenAI-compatible `/v1` API, so Mellea talks to it
through `OpenAIBackend` with a redirected `base_url`. Full script:
[`scratchpad/llama_granite_client.py`](scratchpad/llama_granite_client.py)
(`scratchpad/` is git-ignored — this is a local experiment, not a shipped example).

```python
from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_4_1_30B
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext

backend = OpenAIBackend(
    model_id="ibm-granite/granite-4.1-30b",
    formatter=TemplateFormatter(model_id=IBM_GRANITE_4_1_30B),
    base_url="http://127.0.0.1:8080/v1",
    api_key="llama.cpp",
    model_options={ModelOption.MAX_NEW_TOKENS: 300, ModelOption.TEMPERATURE: 0.0},
)

m = MelleaSession(backend, ctx=ChatContext())

answer = m.instruct(
    "Explain what a GGUF file is in two sentences, for someone who "
    "already knows what a neural network is."
)
print(answer)

followup = m.chat("Now name one trade-off of using a Q4_K_M quantization of that file.")
print(followup.content)
```

Three details that are easy to get wrong:

1. **`model_id` must be a string, not the `ModelIdentifier`.** Passing
   `IBM_GRANITE_4_1_30B` directly to `OpenAIBackend` trips an assertion, because
   `OpenAIBackend` resolves a `ModelIdentifier` via its `openai_name` field and
   the Granite identifiers only define `hf_model_name` / `ollama_name`.
2. **Pass the `ModelIdentifier` to the formatter instead.** `TemplateFormatter`
   uses it to locate Mellea's Granite prompt templates under
   `mellea/templates/prompts/granite/`. The plain string is what goes over the
   wire as the OpenAI `model` field, matching `--alias`.
3. **`api_key` must be non-empty** to satisfy the OpenAI client, but its value
   is ignored unless the server was started with `--api-key`.

Run it:

```bash
cd ~/mellea
uv run python scratchpad/llama_granite_client.py
```

Actual output:

```text
--- instruct ---
A GGUF (Generic GGML Unified Format) file is a standardized file format designed
to store the weights and architecture of neural networks, particularly those
utilizing the GGML library. It enables efficient loading and interoperability of
models across different platforms and frameworks that support the GGUF
specification.

--- chat (same ChatContext) ---
One trade-off of using Q4_K_M quantization in a GGUF file is that while it
significantly reduces the model's memory footprint and speeds up inference by
using lower-precision arithmetic (4-bit quantization), it may also lead to a
decrease in the model's accuracy and expressive power compared to
higher-precision representations.
```

Total wall time was 17 s for both turns. The `ChatContext` carries the first
exchange into the second call, which is why the follow-up resolves "that file"
correctly.

## 6. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `--list-devices` shows no `CUDA0` | CUDA backend not compiled. Re-run `cmake -B build` and confirm `-- Including CUDA backend` appears; check `nvcc --version`. |
| Decode far below ~12 tok/s | Layers likely running on CPU. Confirm `--n-gpu-layers 999` and that the binary is the CUDA build under `build/bin/`. |
| `AssertionError: model_id is None ... no openai_name` | Passed a `ModelIdentifier` to `OpenAIBackend`. Pass the string; give the `ModelIdentifier` to `TemplateFormatter`. |
| `APIConnectionError` from the Mellea script | Server not running, or `base_url` missing the `/v1` suffix. Check `curl http://127.0.0.1:8080/v1/models`. |
| Malformed or empty responses | Server started without `--jinja`, so the GGUF chat template is not applied. |
| `nvidia-smi` shows `[N/A]` memory | Expected on GB10 unified memory. Use `llama-cli --list-devices` or `free -g`. |
| Wrong `model` name rejected by a stricter client | Match the client's `model_id` to `--alias`, or query `/v1/models` for the served name. |

## 7. Reproduce from scratch

```bash
# build
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build ~/llama.cpp/build --config Release -j "$(nproc)"

# model
uv run hf download ibm-granite/granite-4.1-30b-GGUF granite-4.1-30b-Q4_K_M.gguf \
  --local-dir ~/models/granite-4.1-30b-GGUF

# serve
~/llama.cpp/build/bin/llama-server \
  --model ~/models/granite-4.1-30b-GGUF/granite-4.1-30b-Q4_K_M.gguf \
  --alias ibm-granite/granite-4.1-30b \
  --host 127.0.0.1 --port 8080 --n-gpu-layers 999 --ctx-size 8192 --jinja

# query (separate shell)
cd ~/mellea && uv run python scratchpad/llama_granite_client.py
```
