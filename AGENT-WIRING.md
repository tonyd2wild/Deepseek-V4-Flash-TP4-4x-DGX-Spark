# Wiring agents & clients to your DS4 TP=4 endpoint

At TP=4 there is **one** endpoint to point at — the **head node** (rank 0), which serves a
standard **OpenAI-compatible API** on `:8000`. No relay in front of two lanes; it's a single
instance. Anything that can talk to the OpenAI Chat Completions API can talk to this — you
only ever need three things:

1. **Base URL:** `http://<HEAD_NODE_IP>:8000/v1`
2. **Model name:** either served alias — `deepseek-v4-flash-spark` or `deepseek-v4-flash`
   (they resolve to the same model; pick whichever your client already requests).
3. **API key:** the server doesn't authenticate, but most clients *require* a non-empty
   string. Pass any placeholder, e.g. `local` or `sk-no-key`.

> `<HEAD_NODE_IP>` is the head's address on whatever network your clients reach it over
> (LAN, VPN/overlay, etc.) — **not** the RoCE fabric IP (`192.168.192.1`), which is only for
> the inter-node collectives.

---

## Verify with curl

```bash
# list the served models
curl http://<HEAD_NODE_IP>:8000/v1/models

# a chat completion
curl http://<HEAD_NODE_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

If the very first call times out, retry — see the cold-start note in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<HEAD_NODE_IP>:8000/v1",
    api_key="local",  # any non-empty string
)

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Tool/function calling works (`--tool-call-parser deepseek_v4 --enable-auto-tool-choice`) —
pass `tools=[...]` as you would against OpenAI. The reasoning field is parsed via
`--reasoning-parser deepseek_v4` and surfaced in the response.

---

## LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://<HEAD_NODE_IP>:8000/v1",
    api_key="local",
    model="deepseek-v4-flash",
)
print(llm.invoke("Say hello in one sentence.").content)
```

---

## OpenWebUI

Add an **OpenAI-compatible connection**:

- **API Base URL:** `http://<HEAD_NODE_IP>:8000/v1`
- **API Key:** `local` (any non-empty string)

The model(s) appear in the picker by their served-model names.

---

## Generic agent frameworks (Hermes, OpenCode, etc.)

Any framework that lets you define an **OpenAI-compatible provider** works the same way:

```
baseURL: http://<HEAD_NODE_IP>:8000/v1
apiKey:  local             # any non-empty string
model:   deepseek-v4-flash # or the deepseek-v4-flash-spark alias
```

…then point your agent(s) at `<provider>/deepseek-v4-flash`. It's just an OpenAI endpoint.

### Why one TP=4 endpoint is nice for agent fleets

The whole point of the TP=4 lane is that a **single** instance has a **~6M-token KV pool** —
big enough to batch several concurrent agent sessions (we run `--max-num-seqs 6` by default
and it holds usable per-stream speed; see the C1–C6 numbers in the [README](README.md)). So
you get multi-session concurrency **without** standing up two TP=2 lanes and a round-robin
relay in front of them. One base URL, one model name, done.

### Tip: served-model aliases for zero-downtime swaps

The server is launched with multiple `--served-model-name` aliases, so you can later swap
the underlying model (or rebuild the image) and — as long as the new server keeps the same
alias string — every downstream client keeps working without re-pointing.
