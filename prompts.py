"""Benchmark workload: four prompt categories that span the "copy-ability" spectrum.

Speculative decoding only pays off when the draft guesses right. How often it guesses
right depends heavily on the *shape* of the output:

    code_edit   output ~= the prompt's code with a small change  -> n-gram should shine
    extract     output copies spans of the prompt into JSON       -> n-gram should shine
    rag_qa      output quotes/paraphrases a provided context      -> in between
    open_ended  output is new text with no source to copy from    -> n-gram should do little

`build_prompts(n)` interleaves categories round-robin, so any n stays balanced and
the result is deterministic (same n -> same prompts, same order).
"""
from __future__ import annotations

CATEGORIES = ("code_edit", "extract", "rag_qa", "open_ended")

_CODE_EDIT = [
    '''Rename the function `calc` to `compute_invoice_total` everywhere and add type hints.
Return the full updated code only.

```python
def calc(items, tax_rate, discount=None):
    subtotal = 0
    for item in items:
        subtotal += item["price"] * item["qty"]
    if discount is not None:
        subtotal = subtotal - discount
    tax = subtotal * tax_rate
    total = subtotal + tax
    return round(total, 2)


def print_invoice(items, tax_rate):
    total = calc(items, tax_rate)
    for item in items:
        print(f"{item['name']:<20} {item['qty']:>3} x {item['price']:>8.2f}")
    print(f"{'TOTAL':<20} {total:>16.2f}")
```''',
    '''Add input validation to this Java method: throw IllegalArgumentException if `capacity`
is not positive. Return the complete class.

```java
public class LruCache<K, V> {
    private final int capacity;
    private final LinkedHashMap<K, V> map;

    public LruCache(int capacity) {
        this.capacity = capacity;
        this.map = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > LruCache.this.capacity;
            }
        };
    }

    public V get(K key) {
        return map.getOrDefault(key, null);
    }

    public void put(K key, V value) {
        map.put(key, value);
    }
}
```''',
    '''Convert this function from callbacks to async/await. Keep the same names and logging.
Return the full file.

```javascript
const fs = require("fs");

function loadConfig(path, callback) {
  fs.readFile(path, "utf8", (err, data) => {
    if (err) {
      console.error("failed to read config", err);
      return callback(err);
    }
    let parsed;
    try {
      parsed = JSON.parse(data);
    } catch (parseErr) {
      console.error("config is not valid JSON", parseErr);
      return callback(parseErr);
    }
    console.log("config loaded from", path);
    callback(null, parsed);
  });
}

module.exports = { loadConfig };
```''',
    '''Add a docstring to every function below and fix the off-by-one bug in `window_sums`.
Return the full corrected code.

```python
def window_sums(nums, k):
    sums = []
    for i in range(len(nums) - k):
        sums.append(sum(nums[i:i + k]))
    return sums


def max_window(nums, k):
    best = None
    for s in window_sums(nums, k):
        if best is None or s > best:
            best = s
    return best


def moving_average(nums, k):
    return [s / k for s in window_sums(nums, k)]
```''',
]

_EXTRACT = [
    '''Extract every order from this support email as a JSON list of objects with keys
"order_id", "product", "quantity", "ship_to". Copy values exactly as written.

Hi team, following up on my three orders. Order A-10293 was 2 units of the Acme Trail
Backpack 40L shipping to 1458 N Milwaukee Ave, Chicago, IL 60622. Order A-10301 was
1 unit of the Acme Ultralight Tent (2-person) shipping to the same address. Order
A-10322 was 4 units of the Acme Titanium Spork, but that one should go to my office at
233 S Wacker Dr, Suite 4400, Chicago, IL 60606. Two of them have not arrived yet.
Thanks, Dana Whitfield''',
    '''Turn this server log excerpt into JSON: a list of {"timestamp", "level", "service",
"message"} for every line with level WARN or ERROR. Copy the text exactly.

2026-10-12T09:14:02Z INFO  api-gateway  request completed path=/v1/chat status=200 ms=412
2026-10-12T09:14:03Z WARN  scheduler    kv cache usage at 91% (running=48 waiting=12)
2026-10-12T09:14:05Z INFO  api-gateway  request completed path=/v1/chat status=200 ms=388
2026-10-12T09:14:07Z ERROR worker-3     CUDA out of memory while allocating 512.00 MiB
2026-10-12T09:14:07Z WARN  scheduler    preempting request id=chat-8812 (recompute)
2026-10-12T09:14:09Z INFO  worker-3     restarted after OOM, warmup complete
2026-10-12T09:14:11Z ERROR api-gateway  upstream timeout path=/v1/chat ms=30000''',
    '''From this job posting, return JSON with keys "title", "company", "location",
"salary_range", "required_skills" (list), "nice_to_have" (list). Copy phrases verbatim.

Inference Systems Engineer - Northwind AI - Chicago, IL (hybrid). Salary range:
$165,000 - $210,000. Required: production experience with vLLM or TensorRT-LLM, strong
Python and CUDA fundamentals, Kubernetes, Prometheus and Grafana, experience profiling
GPU workloads. Nice to have: speculative decoding, quantization (AWQ, FP8), Triton
kernels, contributions to open-source inference engines.''',
    '''Convert this meeting note into JSON with keys "date", "attendees" (list),
"decisions" (list), "action_items" (list of {"owner", "task", "due"}). Copy wording exactly.

Date: October 21, 2026. Attendees: Priya Raman, Marcus Lee, Ana Souza, Tom Becker.
Decisions: move the chat endpoint to vLLM with prefix caching enabled; keep the
batch endpoint on the old stack until Q1. Action items: Marcus Lee will benchmark
n-gram speculative decoding on the code-assist traffic by October 28. Ana Souza will
add acceptance-rate panels to the Grafana dashboard by October 30. Tom Becker will
draft the rollback runbook by November 4.''',
]

_RAG_QA = [
    '''Answer the question using only the context. Quote the sentences you rely on.

Context: The KV cache stores the key and value tensors for every token already
processed, so the model does not recompute attention inputs for the prefix at each
step. Its size grows linearly with sequence length and batch size. PagedAttention
splits the cache into fixed-size blocks, which removes most fragmentation and lets the
scheduler admit more concurrent requests. When the cache is full, vLLM preempts
requests and later recomputes or swaps them back in.

Question: Why does PagedAttention let a server handle more concurrent requests?''',
    '''Using only the passage, explain what happens during prefill and decode.
Quote the relevant sentences before explaining.

Passage: Prefill processes all prompt tokens in a single forward pass and is usually
compute-bound because many tokens are multiplied against the weights at once. Decode
generates one token per forward pass and is usually memory-bandwidth-bound, because
the full set of weights must be read from GPU memory to produce a single token. This
asymmetry is why time-to-first-token and time-per-output-token are tracked separately.''',
    '''Answer from the context only, citing the exact sentences.

Context: Speculative decoding lets a cheap proposer guess several future tokens. The
target model then scores all guessed positions in one forward pass. Guesses are kept
up to the first disagreement, and the target's own prediction at that position is
appended, so every step yields at least one token. With greedy sampling the output is
identical to running the target alone. The speedup depends on the acceptance rate and
on how cheap the proposer is relative to the target.

Question: Why can speculative decoding never produce fewer than one token per step?''',
    '''Use only this context. Quote it, then answer in two sentences.

Context: Continuous batching inserts new requests into the running batch at every
decode step instead of waiting for the whole batch to finish. This keeps the GPU busy
when requests have very different output lengths. Chunked prefill splits long prompts
into pieces so that a single long prompt does not stall the decode steps of every
other request in the batch.

Question: What problem does chunked prefill solve?''',
]

_OPEN_ENDED = [
    "Write a short story (about 200 words) about a lighthouse keeper who discovers the "
    "light has been signaling to someone across the sea.",
    "Give me five creative names for a coffee shop run by retired astronauts, with a "
    "one-sentence pitch for each.",
    "Explain to a curious 12-year-old why the sky turns orange at sunset. Use an analogy.",
    "Write a persuasive paragraph arguing that every city should have a public tool library.",
]

_BY_CATEGORY = {
    "code_edit": _CODE_EDIT,
    "extract": _EXTRACT,
    "rag_qa": _RAG_QA,
    "open_ended": _OPEN_ENDED,
}


def build_prompts(n: int = 16, categories=CATEGORIES) -> list[tuple[str, str]]:
    """Return n (category, prompt) pairs, round-robin across categories.

    With 4 categories x 4 templates, n=16 uses every template exactly once.
    Larger n cycles through templates again (useful for longer timing runs).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    unknown = set(categories) - set(_BY_CATEGORY)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    out = []
    for i in range(n):
        cat = categories[i % len(categories)]
        templates = _BY_CATEGORY[cat]
        out.append((cat, templates[(i // len(categories)) % len(templates)]))
    return out
