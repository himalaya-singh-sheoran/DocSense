# Lead AI Engineer Interview Guide: LLMs, Agentic AI & Production AI Systems

> Target: Lead / Staff-level AI Engineer at FAANG and Big Tech.
> Assumes: several years of software engineering; hands-on Python, Java, cloud, distributed systems, data pipelines, LLMs, RAG and agentic AI.
> Focus: architecture, production experience, scalability, reliability, security, cost, latency and trade-offs. Not definitions.
> Caveat: model names, framework APIs and prices change quickly. Treat named tools as examples of a *category*, and verify current details before quoting numbers in an interview.

---

## How to Use This Guide

Every question follows the same layout:

| Field | Meaning |
|---|---|
| **Question** | What you will be asked |
| **Interviewer is testing** | The real signal behind the question |
| **Expected answer / key points** | What a strong answer covers |
| **Senior/Lead considerations** | What separates Lead from Senior |
| **Common mistakes** | Answers that lose points |
| **Follow-ups** | Where the interviewer digs next |

**Practice method:** cover everything under the question, answer out loud for 3 to 5 minutes, then compare. Rehearse *with numbers* (latencies, sizes, costs); vague answers read as inexperience.

### What "Lead" means in these interviews

| Signal | Senior answer | Lead answer |
|---|---|---|
| Scope | Solves the stated problem | Reframes the problem, questions requirements, names non-goals |
| Trade-offs | Lists pros and cons | Picks one, states the decision criteria, and says what would make you change your mind |
| Failure | Handles the happy path | Enumerates failure modes, blast radius, degradation strategy |
| Cost | Mentions cost | Estimates cost per request and per tenant, and designs levers to control it |
| Evaluation | "We test it" | Defines metrics, golden sets, online and offline loops, regression gates |
| People | Delivers own work | Sets standards, sequences a roadmap, de-risks with milestones, mentors, aligns stakeholders |
| Reality | Textbook | "In production we saw X, so we did Y" |

### Universal answer framework for design and scenario questions

1. **Clarify**: users, scale (QPS, tokens/request, corpus size), latency SLO, accuracy bar, compliance, budget.
2. **State assumptions and non-goals** explicitly.
3. **Baseline first**: the simplest thing that could work, and why it may not be enough.
4. **High-level architecture**: components and data flow (draw it).
5. **Deep-dive the hardest 2 or 3 parts** (usually retrieval quality, tool safety, serving cost).
6. **Failure modes and degradation**: what breaks, what the user sees, how you recover.
7. **Evaluation and observability**: how you know it works and keeps working.
8. **Security, privacy, compliance.**
9. **Cost and capacity numbers.**
10. **Evolution**: what you would build in v1, v2, v3.

---

## Table of Contents

1. [LLM Fundamentals](#1-llm-fundamentals)
2. [LLM Application Engineering](#2-llm-application-engineering)
3. [Agentic AI](#3-agentic-ai)
4. [Production Hosting & Deployment](#4-production-hosting--deployment)
5. [LLM / Agentic AI System Design](#5-llm--agentic-ai-system-design)
6. [Scalability, Performance & Cost](#6-scalability-performance--cost)
7. [LLM / Agent Evaluation & Observability](#7-llm--agent-evaluation--observability)
8. [Security & Responsible AI](#8-security--responsible-ai)
9. [Scenario-Based Lead Engineer Questions](#9-scenario-based-lead-engineer-questions)
10. [Deep-Dive Follow-Up Ladders](#10-deep-dive-follow-up-ladders)
11. [Coding & Whiteboard Exercises](#11-coding--whiteboard-exercises)
12. [Leadership, Behavioral & Org Questions](#12-leadership-behavioral--org-questions)
13. [Data Engineering for AI](#13-data-engineering-for-ai-pipelines)
14. [Cheat Sheet: Numbers, Formulas, Vocabulary](#14-cheat-sheet-numbers-formulas-vocabulary)
15. [Red Flags & Anti-Patterns Interviewers Listen For](#15-red-flags--anti-patterns-interviewers-listen-for)
16. [Company-Specific Interview Notes](#16-company-specific-interview-notes)
17. [Six-Week Study Plan](#17-six-week-study-plan)
18. [Questions to Ask the Interviewer](#18-questions-to-ask-the-interviewer)

---

# 1. LLM Fundamentals

## Q1.1 Walk through a decoder-only transformer forward pass. Where do compute and memory go during prefill versus decode?

**Interviewer is testing:** Whether you have a mental model of *why* inference behaves the way it does, since every serving optimization derives from it.

**Expected answer / key points**
- Token IDs → embeddings → N blocks of [norm → self-attention (Q/K/V projections, positional encoding such as RoPE, causal mask) → residual → norm → FFN (often SwiGLU) → residual] → final norm → LM head → logits → sampling.
- **Prefill:** all prompt tokens are processed in parallel. Large matrix-matrix multiplies, so it is **compute-bound**. It determines **TTFT**. Attention cost grows O(n²) with prompt length.(Prefill parallelizes computation across all prompt tokens within each layer, while transformer layers themselves execute sequentially.)
- **Decode:** one new token per sequence per step. Each step must read *all model weights* plus the sequence's KV cache from HBM to produce one token, so it is **memory-bandwidth-bound**. It determines **inter-token latency (ITL/TPOT)**.
- Forward FLOPs are roughly 2 × parameters per token. Decode at batch size 1 has very low arithmetic intensity, so GPUs sit mostly idle on compute.
- **Batching amortizes weight reads:** more concurrent sequences per step means more tokens per byte of weights read, which is why continuous batching raises throughput dramatically.
- Rule of thumb for a decode ceiling at batch 1: tokens/s ≈ memory bandwidth ÷ bytes of weights read per token.

**Senior/Lead considerations**
- Tie the model to product levers: long prompts hurt TTFT and prefill cost; long outputs hurt total latency and GPU-seconds; chatty agents multiply both.
- Prefill and decode have different bottlenecks, which is the basis for **disaggregated prefill/decode** serving and **chunked prefill**.
- Rough example: a 70B model in FP16 is about 140 GB of weights. On one accelerator with about 3.35 TB/s bandwidth, the batch-1 ceiling is roughly 24 tokens/s, so you shard across GPUs (tensor parallel) and batch requests.

**Common mistakes**
- Saying "GPU compute is the bottleneck" for decode.
- Ignoring the KV cache as a memory consumer.
- Treating all tokens (input and output) as equal cost. They differ in latency profile and, usually, price.

**Follow-ups**
- Why does doubling batch size not double ITL?
- What happens to throughput as batch grows, and what limits it? (KV cache memory, then compute.)
- How does tensor parallelism change the compute/memory picture, and what does it cost? (All-reduce over NVLink/interconnect.)
- Why is prefill/decode disaggregation attractive, and what is the new problem it creates? (KV cache transfer between pools.)

---

## Q1.2 Explain MHA, MQA, GQA and MLA. Calculate the KV cache for a 70B-class model and explain why it matters.

**Interviewer is testing:** Capacity-planning ability grounded in architecture.

**Expected answer / key points**
- MHA gives every query head its own K/V, providing maximum attention capacity but the largest KV cache. MQA shares one K/V across all query heads, minimizing the cache but potentially reducing quality. GQA groups query heads and shares K/V within each group, giving a practical trade-off between quality and memory. MLA goes further by compressing KV information into a latent representation, substantially reducing KV-cache memory and inference bandwidth.
- KV cache per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element.
- Example (Llama-3-70B-class: 80 layers, 8 KV heads, head_dim 128, FP16): 2 × 80 × 8 × 128 × 2 B ≈ **320 KB/token**. An 8k-token sequence is about **2.6 GB**; 32 concurrent 8k sequences is about 84 GB of cache, on top of about 140 GB of weights.
- The cache, not weights, often limits **concurrency** at long context.
- Mitigations: GQA/MQA/MLA, KV cache quantization (FP8/INT8), PagedAttention (removes fragmentation), prefix sharing, sliding-window or sparse attention, offloading to CPU/NVMe, eviction/compression.

**Senior/Lead considerations**
- Concurrency ≈ (GPU memory − weights − activations/overhead) ÷ (per-sequence KV). Use this to size clusters and to justify context-length limits or tiered pricing.
- Model choice affects serving cost: a GQA model can serve materially more concurrent users per GPU than an equivalent MHA model.
- Long-context requests are *expensive tenants*: cap, meter, or route them separately.

**Common mistakes**
- Forgetting the ×2 (K and V) or the layer count.
- Using attention heads instead of KV heads for GQA models.
- Assuming a bigger context window is free.

**Follow-ups**
- How does PagedAttention improve utilization? (Fixed-size blocks, block table, little internal fragmentation, enables copy-on-write sharing.)
- What is FlashAttention and does it change the KV cache size? (It is an IO-aware exact algorithm that avoids materializing the n×n matrix; it does not shrink the cache.)
- What is the quality trade-off of KV cache quantization?

---

## Q1.3 What actually limits usable context length, and when is long-context better than RAG (or worse)?

**Interviewer is testing:** Judgment about a very common product decision.

**Expected answer / key points**
- Limits: O(n²) attention compute in prefill, KV memory growth, positional generalization (RoPE scaling, YaRN and similar), and **effective** context being smaller than advertised ("lost in the middle", degraded multi-hop reasoning over long inputs).
- Long-context wins: small-to-medium bounded corpora, cross-document reasoning, one-off analyses, when retrieval recall would be poor (needs holistic view), rapid prototyping.
- RAG wins: large or changing corpora, per-user access control, cost and latency sensitivity, need for citations/provenance, freshness, auditability.
- Prompt caching narrows the gap for repeated large contexts (pay full price once, then discount on cache hits).
- Hybrid: retrieve broadly, rerank, then pass a moderately sized, well-ordered context.

**Senior/Lead considerations**
- Cost math: 100k input tokens per query at 1,000 QPS is a very different bill from 4k retrieved tokens. Quantify.
- Evaluate with **needle-in-haystack AND task-realistic tests**; needle tests overstate real capability.
- Ordering matters: put critical evidence near the start or end; keep instructions stable at the start to maximize cache hits.

**Common mistakes**
- "Long context makes RAG obsolete."
- Trusting the advertised window as the reliable window.
- Ignoring that ACLs are much easier to enforce at retrieval time than after stuffing everything into a prompt.

**Follow-ups**
- How would you measure the effective context of a model for your task?
- How do you decide the retrieval `k` versus stuffing everything?
- What breaks if the corpus doubles?

---

## Q1.4 Tokenization and embeddings: what production problems do they cause, and how do you choose and migrate an embedding model?

**Interviewer is testing:** Practical depth beyond "BPE splits words."

**Expected answer / key points**
- Subword tokenizers (BPE, SentencePiece, unigram) trade vocabulary size against sequence length. Issues: numbers and code tokenize poorly; non-English text can cost 2 to 4× more tokens; whitespace and formatting change token counts; different providers count differently (cost estimation must use the provider's tokenizer).
- Token limits, truncation and chunk sizes must be computed in **tokens**, not characters.
- Embedding model selection: retrieval quality on **your** domain (build a labeled eval set), dimensionality (index size and latency), max input length, multilingual needs, licensing, hosting/latency, and cost. Matryoshka-style embeddings(Matryoshka embeddings are embeddings trained such that prefixes of the embedding remain semantically meaningful. For example, a 768-dimensional embedding can be truncated to 128, 256, or 512 dimensions while retaining useful retrieval quality. This allows us to trade off vector-search cost and accuracy dynamically without maintaining separate embedding models. In a large RAG system, we can use a smaller prefix for initial retrieval and a larger representation or reranker for more accurate ranking.) allow truncating dimensions.
- **Migration:** vectors from different models are not comparable. Re-embed the whole corpus, run dual indexes (blue/green), shadow-evaluate, then cut over; keep model version in index metadata.
- Similarity: cosine or inner product on normalized vectors; ANN index (HNSW, IVF-PQ) trades recall for latency and memory.

**Senior/Lead considerations**
- Embedding-model change is an *infra migration*, not a config change: budget re-embedding cost/time and plan rollback.
- Domain-specific and query-vs-document asymmetry (instruction-prefixed models) matter; test them.
- Consider fine-tuning embeddings or adding a reranker before switching to a bigger model.

**Common mistakes**
- Picking a model by leaderboard rank without a domain eval.
- Mixing embeddings from two model versions in one index.
- Chunking by characters.

**Follow-ups**
- How would you re-embed 500M chunks with zero downtime?
- Quantized or binary embeddings: what do you gain and lose?
- Why might recall@k look great offline yet answers still fail?

---

## Q1.5 Pre-training, instruction tuning, RLHF/DPO and RL with verifiable rewards: what does each stage do, and when should you fine-tune versus prompt versus RAG?

**Interviewer is testing:** Ability to make build-vs-tune decisions with cost awareness.

**Expected answer / key points**
- **Pre-training:** next-token prediction on huge corpora; gives knowledge and capability. **SFT / instruction tuning:** teaches format and instruction-following from curated demonstrations(basically format the input like User: Model:). **Preference optimization (RLHF( Reinforcement Learning with Human Feedback) with PPO (Proximal Policy Optimization)-style methods, DPO and variants, RLAIF):** aligns to human or AI preferences (helpfulness, safety, style). **RL with verifiable rewards:** trains reasoning and tool-use behavior where outcomes can be checked automatically (math, code, tests).
- Decision ladder: **prompting → few-shot/structured outputs → RAG → tool use → fine-tuning → continued pre-training**. Move down only when the cheaper option's measured ceiling is insufficient.
- Fine-tune for: style/format consistency, domain-specific behaviors, latency/cost reduction via a smaller distilled model, tool-calling reliability. Do **not** fine-tune to inject frequently changing facts (use RAG).
- Distillation: train a small model on a large model's outputs to cut serving cost.

**Senior/Lead considerations**
- Fine-tuning creates an ownership burden: data pipeline, eval, re-tuning when the base model changes, and safety regression testing.
- Catastrophic forgetting and alignment drift after SFT need regression suites.
- A fine-tuned 8B model can beat a frontier model on a narrow task at a fraction of the cost, but only with good data and evals.

**Common mistakes**
- Fine-tuning to add knowledge.
- No held-out eval or baseline comparison against a well-prompted large model.
- Data quality neglected (duplicates, label noise, contamination).

**Follow-ups**
- How much data do you need, and how do you build it? (Curated quality > quantity; synthetic data with verification.)
- How do you detect that a fine-tune regressed safety behavior?
- When is distillation better than fine-tuning a general model?

---

## Q1.6 Explain LoRA, QLoRA and other PEFT techniques, including hyperparameters and how you would serve many adapters.

**Interviewer is testing:** Practical fine-tuning and multi-tenant serving knowledge.

**Expected answer / key points**
- **LoRA:** freeze the base weights W; learn a low-rank update ΔW = (α/r)·B·A with rank r ≪ d. Trainable parameters per matrix drop from d×k to r(d+k). No added inference latency if merged.
- **QLoRA:** base model quantized to 4-bit (NF4), with double quantization and paged optimizers; adapters trained in higher precision. Enables fine-tuning large models on limited GPUs.
- Others: adapters, prefix/prompt tuning, IA³, DoRA (weight decomposition), full fine-tuning with ZeRO/FSDP.
- Hyperparameters: rank r, alpha, dropout, target modules (attention only vs attention + MLP), learning rate, epochs. Higher rank means more capacity but more overfitting/memory.
- **Multi-adapter serving:** keep one base model in memory and hot-swap or batch across many LoRA adapters per request (supported by modern serving engines), giving per-tenant customization with shared GPUs.

**Senior/Lead considerations**
- Multi-tenant fine-tuning is a cost story: one GPU pool serves hundreds of adapters instead of hundreds of full models.
- Adapter versioning, evaluation gates and rollback matter as much as training.
- LoRA typically matches full fine-tuning on narrow tasks but can lag on large distribution shifts.

**Common mistakes**
- Merging adapters into a quantized base without checking quality.
- Treating QLoRA as identical to full precision quality at inference.
- No per-adapter eval.

**Follow-ups**
- How do adapters interact with quantized serving?
- How do you prevent one tenant's data from leaking into another's adapter? (Isolated training data, no shared adapters, audit.)
- When would you choose full fine-tuning?

---

## Q1.7 Quantization: what are the options, and what can go wrong?

**Interviewer is testing:** Ability to trade quality for cost with measurement.

**Expected answer / key points**
- **Weight-only** quantization (INT8/INT4 via GPTQ, AWQ) reduces memory and bandwidth; large decode gains because decode is bandwidth-bound. **Weight+activation** (SmoothQuant, FP8 on supporting hardware) speeds compute as well.
- **KV cache quantization** raises concurrency.
- PTQ (post-training, cheap, calibration data) versus QAT (better quality, expensive).
- Memory: 70B params ≈ 140 GB FP16, about 70 GB INT8, about 35 GB INT4 (plus overhead).
- Risks: quality loss concentrated in reasoning, math, long context, low-resource languages, and outlier-heavy layers; kernel support and speedups vary by hardware; batch-size-dependent performance (weight-only INT4 helps most at small batch).

**Senior/Lead considerations**
- Always evaluate on **your task distribution**, including safety and tool-calling accuracy, not just perplexity.
- Quantization can reduce GPU count per replica (for example 2 GPUs to 1), which changes availability and packing economics.
- Roll out behind a shadow or canary comparison.

**Common mistakes**
- Judging by perplexity only.
- Ignoring the calibration set's mismatch with production data.
- Assuming INT4 is always faster (kernel and batch dependent).

**Follow-ups**
- Why is weight-only quantization especially effective for decode?
- What breaks in tool-calling or JSON adherence after aggressive quantization?
- FP8 versus INT8: when and why?

---

## Q1.8 List and explain the main inference-optimization techniques, and how you would decide which to apply first.

**Interviewer is testing:** Breadth plus prioritization.

**Expected answer / key points**
- **Continuous (in-flight) batching:** schedule at iteration level so finished sequences free slots immediately.
- **PagedAttention:** block-based KV memory management (PagedAttention is a technique for managing the Transformer KV cache using fixed-size blocks, inspired by virtual memory paging. Instead of allocating one contiguous KV-cache region per request, logical token blocks are mapped to physical GPU blocks through a block table. This reduces memory fragmentation and over-allocation, allows dynamic allocation as sequences grow, and enables sharing of common KV-cache prefixes. It doesn't change the attention mathematics; it optimizes KV-cache memory management and access, which is particularly important for high-concurrency LLM serving.)
- **Prefix/prompt caching:** reuse KV for shared prefixes (system prompts, few-shot, documents); radix-tree style caches.
- **Speculative decoding:** draft proposes k tokens, target verifies in one pass, output distribution preserved via rejection sampling; best at low batch, latency-sensitive workloads; gain depends on acceptance rate.
- **Chunked prefill:** interleave prefill chunks with decode to protect ITL.
- **Quantization**, **tensor/pipeline/expert parallelism**, **FlashAttention**(FlashAttention is an IO-aware, tiled implementation of exact Transformer attention. Standard attention materializes the \(N\times N\) attention matrix, causing substantial GPU HBM reads and writes. FlashAttention computes attention in blocks, keeps intermediate results in fast on-chip SRAM, and uses an online softmax algorithm so it doesn't need to materialize the full attention matrix. It therefore reduces memory usage from quadratic intermediate storage to roughly linear in sequence length and significantly improves attention performance, while preserving the attention computation itself.), kernel fusion, CUDA graphs.
- **Prefill/decode disaggregation**, **MoE** expert parallelism, **model routing/cascades**, **distillation**.
- Prioritize by workload profile: measure first (profile TTFT, ITL, queueing, GPU utilization, cache hit rate), then apply the cheapest high-impact levers (engine choice with continuous batching + prefix caching, quantization), then heavier ones.

**Senior/Lead considerations**
- Optimize **goodput** (requests meeting SLO per GPU-hour), not raw tokens/s.
- Some techniques conflict (for example speculative decoding gains shrink at high batch).
- Engineering cost and operational complexity are part of the trade-off.

**Common mistakes**
- Listing techniques without a measurement plan.
- Optimizing throughput at the cost of p99 latency.
- Ignoring queueing delay, which frequently dominates TTFT under load.

**Follow-ups**
- What is your first move if p99 TTFT doubles under load?
- How does chunked prefill trade TTFT versus ITL?
- When does speculative decoding hurt?

---

## Q1.9 Why is temperature 0 still non-deterministic, and how do you build reproducible LLM behavior?

**Interviewer is testing:** Debugging maturity and testing strategy.

**Expected answer / key points**
- Causes: floating-point non-associativity with different reduction orders, batch-size-dependent kernels, dynamic batching with other requests, MoE routing ties, hardware/driver/version changes, provider-side model updates, tie-breaking in near-equal logits.
- Sampling parameters: temperature, top-p, top-k, min-p, repetition/presence penalties, seed (where offered; typically "best effort").
- Reproducibility strategy: pin model versions/snapshots, log full prompts and parameters, use seeds where available, design tests as **statistical** (run N samples, assert on pass rate) rather than exact-match, use structured outputs and deterministic tools around the LLM.

**Senior/Lead considerations**
- Treat non-determinism as a design constraint: idempotent tool calls, validation layers, retry-with-critic, and evaluation with confidence intervals.
- Model version drift is an operational risk; contract-test on upgrades.

**Common mistakes**
- Exact-match assertions on free text.
- Assuming a seed guarantees identical outputs across infrastructure.

**Follow-ups**
- How do you regression-test after a provider silently updates a model?
- What sampling settings do you use for extraction versus creative tasks, and why?

---

## Q1.10 Where does hallucination come from, and how do you reduce it in production?

**Interviewer is testing:** Systems approach to a fuzzy problem.

**Expected answer / key points**
- Causes: the training objective rewards fluent continuation; knowledge gaps and stale knowledge; weak calibration and training incentives that reward confident guessing; retrieval failures (irrelevant or missing context); conflicting context; sycophancy; over-long or noisy context; ambiguous prompts.
- Distinguish: **intrinsic** (contradicts provided context) versus **extrinsic** (unsupported by context or world).
- Mitigation layers:
  1. **Grounding:** RAG with quality retrieval, citations, "answer only from context", permission to say "I don't know".
  2. **Tools:** calculators, databases, code execution instead of recall.
  3. **Constrained generation:** schema/grammar-constrained outputs, enumerations.
  4. **Verification:** claim-level entailment checks against sources (NLI or LLM judge), self-consistency, cross-model checking for high-stakes cases.
  5. **UX:** show sources, confidence, allow feedback.
  6. **Evaluation:** groundedness/faithfulness metrics, red-team sets, online monitoring.
  7. **Abstention policy:** thresholds tuned to cost of error.

**Senior/Lead considerations**
- Define "acceptable hallucination rate" per use case (legal and medical differ from brainstorming) and design the system to that bar.
- Retrieval quality is usually the dominant lever in RAG; fix retrieval before blaming the model.
- Verifiers add latency and cost; apply them selectively based on risk.

**Common mistakes**
- "Set temperature to 0."
- Assuming citations imply correctness (citations can be fabricated or irrelevant; verify them).
- Using an LLM judge with no calibration against human labels.

**Follow-ups**
- How do you measure faithfulness at scale and validate your judge?
- What would you do when the retriever returns nothing relevant?
- How do you handle conflicting sources?

---

## Q1.11 Mixture-of-Experts: how does it change serving and cost?

**Interviewer is testing:** Understanding of modern model architectures' operational implications.

**Expected answer / key points**
- MoE replaces the dense FFN with many expert FFNs; a router activates top-k per token. Total parameters are large while **active** parameters per token are small, so compute per token is lower than an equal-quality dense model.
- Memory footprint is still the **total** parameters (all experts must be resident), so you need many GPUs even though FLOPs are low.
- Serving needs **expert parallelism** plus all-to-all communication; load imbalance across experts causes stragglers; batch composition affects expert utilization.
- Good at high throughput with large batches; harder at small batch and low latency because of memory and communication.

**Senior/Lead considerations**
- MoE shifts the bottleneck toward interconnect bandwidth and memory capacity; cluster topology matters.
- Determinism and routing ties add non-determinism.

**Common mistakes**
- "MoE is cheaper to host" (it is cheaper per token, not per deployment footprint).
- Ignoring communication overhead.

**Follow-ups**
- How would you shard a large MoE across 8 GPUs versus 2 nodes?
- What is expert load-balancing loss and why does it exist?

---

# 2. LLM Application Engineering

## Q2.1 How do you engineer prompts for a production system (not a demo)?

**Interviewer is testing:** Treating prompts as versioned, tested software artifacts.

**Expected answer / key points**
- Structure: role/task, constraints, context, output format, examples. Keep the **stable prefix** first (system prompt, tool definitions, few-shot) and volatile content last to maximize prompt-cache hits.
- Prompts live in version control or a prompt registry with IDs, changelogs, owners, environments (dev/stage/prod), and A/B or canary rollout.
- Every prompt change runs through an **eval suite** (golden set, edge cases, adversarial cases) as a CI gate.
- Use templates with typed variables; sanitize and delimit untrusted inputs; never build prompts by naive string concatenation with user data.
- Techniques: few-shot examples chosen dynamically, chain-of-thought or reasoning modes where they pay off, decomposition into smaller steps, self-critique, structured outputs.
- Model-specific tuning: a prompt tuned for one model often regresses on another; keep an abstraction layer and per-model prompt variants.

**Senior/Lead considerations**
- Prompt length is a cost and latency lever; prune, compress and cache.
- Separate **instructions** from **data** (helps security and quality).
- Own the migration path when models are deprecated; contract tests protect you.

**Common mistakes**
- Prompts edited in production with no versioning or evals.
- Long "mega-prompts" with conflicting instructions.
- Optimizing on five hand-picked examples.

**Follow-ups**
- How do you roll out a prompt change to 10% of traffic and decide to promote?
- How do you test prompts across model upgrades?
- When do you break one prompt into a pipeline?

---

## Q2.2 Structured outputs and function/tool calling: how do you make them reliable?

**Interviewer is testing:** Practical reliability engineering around probabilistic components.

**Expected answer / key points**
- Options ranked by strength: **constrained decoding** (grammar/JSON-schema enforced at token level, guaranteeing syntactic validity), provider-native structured output/tool-calling modes, prompt-only JSON with validation and retry.
- Schema design: small, flat, well-described fields; enums over free text; required versus optional; descriptions act as prompts; avoid ambiguous overlapping tools.
- Validation: schema validation (Pydantic/JSON Schema/Java bean validation), **semantic validation** (business rules, referential checks, range checks), then retry with the validation error fed back (bounded attempts), then fallback/escalation.
- Tool-calling flow: model emits call → your runtime validates → executes with **authorization of the end user** → returns result → model continues. The model never executes anything itself.
- Parallel tool calls, forced tool choice, and tool-result truncation/summarization to protect context.

**Senior/Lead considerations**
- Syntactic validity does not equal semantic correctness; a perfectly valid JSON can carry a wrong amount or target.
- Too many tools degrade selection accuracy; route or group tools, or retrieve tool definitions dynamically.
- Idempotency, timeouts, retries and side-effect classification (read/write/destructive) belong in the tool layer.

**Common mistakes**
- Trusting model-provided arguments (IDs, SQL, URLs) without validation and authorization.
- Unbounded retry loops.
- Dozens of overlapping tools with vague descriptions.

**Follow-ups**
- How do you handle a tool result of 200k tokens?
- What is your strategy for tools that are slow (30 s) or flaky?
- How would you evaluate tool selection accuracy?

---

## Q2.3 Design a RAG architecture end to end. What are the components and where does quality get lost?

**Interviewer is testing:** Complete-pipeline thinking; knowing that most RAG failures are retrieval and data failures.

**Expected answer / key points**
- **Ingestion:** connectors → parsing (layout-aware for PDFs/tables/images/OCR) → cleaning and deduplication → chunking → metadata enrichment (source, ACLs, timestamps, tenant, doc type) → embedding → index (vector + keyword) with versioning.
- **Query path:** query understanding (rewrite, decomposition, intent routing, filters) → retrieval (hybrid dense + BM25, metadata filters, ACL filter) → rerank (cross-encoder) → context assembly (dedupe, ordering, token budget, citations) → generation → post-validation (grounding check, citation check) → response with sources.
- **Quality loss points:** parsing errors, bad chunk boundaries, embedding mismatch, missing metadata filters, low recall in top-k, reranker missing, context overstuffing, weak grounding prompts, stale index.
- **Ops:** incremental indexing, deletes and permission changes propagating, re-index/migration, eval and monitoring per stage.

**Senior/Lead considerations**
- Evaluate **stage by stage** (retrieval recall@k, rerank nDCG, faithfulness, answer relevance) so you know where to invest.
- Multi-tenancy and ACLs are first-class: enforce at retrieval, not only at display.
- Start with a baseline (hybrid + rerank + good chunking) before GraphRAG or agentic retrieval; add complexity only where evals show gains.

**Common mistakes**
- Pure vector search with no keyword component (misses IDs, acronyms, exact terms).
- Chunking blindly at fixed sizes.
- No retrieval evaluation, only end-to-end "vibes".

**Follow-ups**
- Answers are wrong for questions requiring several documents; what do you change?
- How do you handle tables, charts and scanned PDFs?
- How do document deletions and permission revocations reach the index, and how fast?

---

## Q2.4 Chunking, embeddings, hybrid retrieval and reranking: what choices matter and how do you tune them?

**Interviewer is testing:** Depth on the components that determine RAG quality.

**Expected answer / key points**
- **Chunking:** semantic/structural boundaries (headings, paragraphs, code blocks, table rows) with overlap; size trade-off (small = precise but context-poor, large = noisy and costly). Techniques: parent-child (retrieve small, return larger parent), sentence-window, contextual chunk headers, **contextual retrieval** (prepend an LLM-generated context blurb per chunk before embedding), late chunking.
- **Retrieval:** dense for semantics, BM25/sparse for exact matches; combine with **reciprocal rank fusion** or learned weights. Query rewriting, multi-query, **HyDE**, step-back, decomposition for multi-hop.
- **ANN indexes:** HNSW (high recall, memory-heavy), IVF-PQ (memory-efficient, lower recall), DiskANN-style for very large scale; tune `ef_search`/`nprobe` for recall versus latency; quantized vectors to cut memory.
- **Reranking:** retrieve 50 to 200 candidates cheaply, then rerank with a cross-encoder or LLM to pick the top 5 to 10. Usually the single best quality-per-effort improvement.
- Tune with a labeled set: recall@k, MRR, nDCG; ablate each change.

**Senior/Lead considerations**
- Reranker latency and cost per query must fit the SLO; cache and batch, or distill.
- Metadata filtering with ANN can hurt recall (pre-filter vs post-filter behavior); test on real filter selectivity.
- Multilingual and domain jargon often need custom embeddings or query expansion.

**Common mistakes**
- One chunk size for all document types.
- Increasing k to "fix" recall without reranking, which floods the context.
- Never testing filtered-search recall.

**Follow-ups**
- Why can a highly selective filter break HNSW recall?
- How do you evaluate chunking strategies objectively?
- When would you add GraphRAG or a knowledge graph?

---

## Q2.5 How do you manage context and memory for long conversations and long-running tasks?

**Interviewer is testing:** Practical strategies for finite context.

**Expected answer / key points**
- **Short-term:** sliding window, summarization (rolling summary), selective retention by importance, tool-result truncation/compaction, offloading big artifacts to files/stores and passing references.
- **Long-term:** external memory in a store (vector + structured), written by explicit "remember" actions or extraction pipelines, read via retrieval keyed by user/task/recency/importance.
- Memory types: episodic (events), semantic (facts), procedural (learned instructions/skills).
- Hygiene: dedupe, decay, conflict resolution (newer beats older?), user-visible and editable memory, deletion on request, PII policy, tenant isolation.
- Context budget: allocate tokens to system prompt, tools, memory, retrieved docs, history, and reserved output.

**Senior/Lead considerations**
- Summaries lose detail and can compound errors; keep raw logs retrievable and summarize with task-specific schemas.
- Memory is an **attack surface** (poisoned memories persist); validate what gets written and provenance-tag it.
- Compaction changes cache prefixes; design so the stable prefix survives.

**Common mistakes**
- Appending everything until the window overflows.
- Storing unverified model-generated "facts" as truth.
- No deletion path (privacy and compliance issue).

**Follow-ups**
- How do you decide what to remember?
- How do you keep memory from going stale or contradicting itself?
- How would you build per-user memory for millions of users cheaply?

---

## Q2.6 Caching for LLM systems: which layers exist and what are the risks?

**Interviewer is testing:** Cost/latency engineering with correctness awareness.

**Expected answer / key points**
- **Prompt/prefix caching** (provider or engine level): cache KV for a repeated prefix; requires stable prefix ordering; large savings on input tokens and TTFT.
- **Exact-match response cache** keyed on normalized request + model + params: safe for deterministic, non-personalized queries.
- **Semantic cache:** embed the query, return a cached answer above a similarity threshold. Cheap and fast but risky (near-duplicate questions with different intent; stale answers; cross-tenant leakage).
- **Retrieval cache**, **embedding cache**, **tool-result cache** (with TTL and invalidation).
- Controls: tenant and permission scoping in cache keys, TTLs, invalidation on source updates, threshold tuning on a labeled set, bypass for personalized or time-sensitive queries.

**Senior/Lead considerations**
- Measure hit rate and **quality-adjusted** savings; a wrong cached answer costs more than a fresh call.
- Semantic caches must never cross authorization boundaries.
- Put cache-friendly structure into prompt design from day one.

**Common mistakes**
- A global semantic cache with a loose threshold.
- Caching answers derived from ACL-protected documents under a shared key.
- Ignoring invalidation.

**Follow-ups**
- How would you choose the similarity threshold?
- How do you invalidate cached answers when a source document changes?
- Prompt cache hit rate dropped after a deploy; why might that be?

---

## Q2.7 Design an LLM gateway with model routing.

**Interviewer is testing:** Platform thinking for multi-model, multi-team environments.

**Expected answer / key points**
- **Functions:** unified API (OpenAI-compatible or internal), authentication and per-team/tenant quotas, rate limiting (RPM and TPM), routing, retries with backoff, timeouts, fallbacks across providers/regions, streaming passthrough, caching, guardrails hooks (PII redaction, moderation), logging/tracing, token and cost accounting, budget enforcement, prompt/version management, model catalog and deprecation management.
- **Routing strategies:** static rules (task to model), cost/latency-based, capability-based, **cascade** (try cheap model, escalate on low confidence or failed validation), **learned router** (classifier predicting difficulty), A/B or shadow routing for evaluation, health-based failover.
- Provider concerns: differing APIs, tokenizers, context limits, tool-calling formats, safety behaviors, and data-residency/contract constraints.
- Data plane must be low-overhead (a few ms), horizontally scalable, and stateless where possible; control plane manages config.

**Senior/Lead considerations**
- The gateway becomes a **critical dependency**; design for HA, config safety (staged rollout, validation), and graceful degradation.
- Fallbacks must respect **data residency and compliance**, not just availability.
- Router quality needs continuous evaluation; a bad router silently degrades quality or inflates cost.

**Common mistakes**
- Fallback to a model with a different safety/behavior profile without evaluation.
- Counting requests instead of tokens for rate limits.
- Gateway that buffers whole responses, breaking streaming.

**Follow-ups**
- How do you implement token-based rate limiting with streaming, where output tokens are unknown up front?
- How do you avoid retry storms during a provider outage?
- How do you evaluate that the cheap-model cascade is not hurting quality?

---

## Q2.8 Guardrails and validation: what do you put where?

**Interviewer is testing:** Defense-in-depth for input, output and actions.

**Expected answer / key points**
- **Input:** authentication/authorization, size limits, PII detection/redaction, prompt-injection and jailbreak detection, topic/policy classification, rate limits.
- **Output:** schema validation, moderation/policy classifiers, PII/secret leakage detection, groundedness and citation checks, format checks, brand/tone rules.
- **Action:** tool allowlists, argument validation, permission checks per user, approval gates for risky actions, dry-run modes, spend/step limits.
- Implementation: layered fast checks (regex, classifiers) before slow checks (LLM judges); run independent checks in parallel; fail closed for high-risk paths; log decisions for audit.
- Guardrails are probabilistic; **hard controls** (authz, sandboxing, least privilege) are what provide guarantees.

**Senior/Lead considerations**
- Every guardrail has latency, cost and false-positive rates; measure them and tune per use case.
- Do not use the same model that may be compromised to police itself as the only defense.
- Keep policy as configuration with versioning and tests.

**Common mistakes**
- Relying on "please don't reveal secrets" in the system prompt.
- Guardrails only on input, not on tool outputs and retrieved content.
- No metrics for false positives, causing users to be blocked silently.

**Follow-ups**
- Your guardrail adds 400 ms; how do you reduce that?
- How do you evaluate a moderation classifier on your domain?
- What do you do on a guardrail block: refuse, redact, or escalate?

---

## Q2.9 How do you balance cost, latency and quality in an LLM feature?

**Interviewer is testing:** Ability to make and defend trade-offs with numbers.

**Expected answer / key points**
- Define the SLO triple: quality bar (eval metric), latency (TTFT and end-to-end), cost per request or per user. Find the **Pareto frontier** by experiment.
- Levers: model size/tier, cascades and routing, prompt shortening, caching, RAG with fewer/better chunks, output length limits, structured outputs to avoid rambling, batching for offline work, streaming to improve perceived latency, quantized or distilled self-hosted models, async processing for non-interactive tasks, parallelizing independent calls.
- Different tiers of work: interactive (low latency) versus batch (cheap, can use discounted batch APIs or spare capacity).
- Compute unit economics: cost per successful task, not per token.

**Senior/Lead considerations**
- Expose cost as a first-class metric to product owners; tie it to feature value.
- Hidden costs: retries, agent loops, long contexts, guardrail/judge calls, embeddings, reranking, storage, observability.
- Budget guards prevent runaway spend (per-request, per-user, per-tenant).

**Common mistakes**
- Optimizing token price while ignoring retry/loop multipliers.
- Choosing the biggest model "to be safe" with no evidence.
- Ignoring perceived latency (streaming, progress UX).

**Follow-ups**
- The CFO says cut LLM spend by 40%. What is your plan and order of operations?
- How do you decide when a smaller model is "good enough"?
- How do you forecast spend for next quarter?

---

## Q2.10 Multi-tenant and permission-aware RAG: how do you prevent one user seeing another's data?

**Interviewer is testing:** Enterprise-grade access-control design.

**Expected answer / key points**
- Propagate **end-user identity** through the pipeline; retrieval must filter by ACLs at query time (document-level, sometimes chunk/field-level).
- Options: per-tenant indexes/namespaces (strong isolation, more overhead), shared index with tenant/ACL metadata filters (efficient, needs rigorous testing), hybrid by tenant size/sensitivity.
- Sync ACLs from source systems (identity provider groups, document permissions) with bounded propagation delay; handle revocation and deletion.
- Isolation must extend to **caches, memory, logs, embeddings for fine-tuning, and traces**.
- Test: adversarial cross-tenant tests in CI; canary documents; audit logs of what was retrieved for whom.

**Senior/Lead considerations**
- Post-filtering after ANN can leak counts/timing and hurt recall; prefer pre-filtering or per-tenant partitions for sensitive data.
- Service-account retrieval with broad access plus "the model will behave" is an anti-pattern.
- Compliance: data residency, retention, right-to-delete flowing into vector stores and backups.

**Common mistakes**
- Filtering only in the UI or in the prompt.
- Forgetting the semantic cache and conversation memory.
- Slow ACL sync leaving revoked access open.

**Follow-ups**
- User loses access to a document at 10:00; when must answers stop citing it?
- How do you test isolation continuously?
- Millions of ACL groups: how does filtering stay fast?

---

## Q2.11 Advanced RAG variants: when would you use agentic RAG, GraphRAG, or multimodal RAG?

**Interviewer is testing:** Knowing when complexity pays off.

**Expected answer / key points**
- **Agentic RAG:** the model decides whether/what/when to retrieve, iterates, reformulates queries, uses multiple tools/sources. Good for multi-hop, ambiguous and multi-source questions; costs more latency, tokens and variance.
- **GraphRAG / knowledge graphs:** entity/relation extraction, community summaries; good for global "summarize themes across the corpus" and relationship-heavy questions; expensive to build and maintain.
- **Multimodal RAG:** images/tables/charts via multimodal embeddings or captioning/OCR; layout-aware parsing.
- **Text-to-SQL / structured retrieval** for analytics questions: schema retrieval, constrained SQL generation, read-only execution, result validation.
- Choose by failure analysis: build a baseline, categorize failures, add the technique that addresses the top failure class.

**Senior/Lead considerations**
- Maintain a routing layer: simple questions take the cheap path.
- Graph extraction quality and freshness are the real cost.
- For SQL: enforce read-only, row-level security, query cost limits, and validated schemas.

**Common mistakes**
- Adopting GraphRAG because it is fashionable.
- Letting agentic retrieval loop without budgets.

**Follow-ups**
- How would you evaluate whether agentic RAG beats the baseline?
- How do you secure LLM-generated SQL?
- How do you keep a knowledge graph fresh?

---

# 3. Agentic AI

## Q3.1 What makes a system an "agent", and when should you NOT build one?

**Interviewer is testing:** Judgment. Strong candidates resist unnecessary agent complexity.

**Expected answer / key points**
- An agent uses an LLM to **decide control flow**: it chooses actions (tools), observes results, and iterates toward a goal with some autonomy. A **workflow** has predefined code paths where LLMs fill steps.
- Spectrum: single LLM call → prompt chain → routing → parallelization → orchestrator-workers → evaluator-optimizer → autonomous agent loop.
- Build an agent when: the path cannot be predetermined, the task is open-ended, steps depend on intermediate results, and the value justifies higher latency, cost and variance.
- Do **not** when a deterministic workflow or a single call meets the bar, when latency/cost budgets are tight, when errors are unrecoverable, or when you cannot evaluate it.
- Start with the simplest architecture and increase autonomy only as evals justify.

**Senior/Lead considerations**
- Autonomy is a dial: define per-action autonomy levels (auto, notify, approve).
- Reliability compounds multiplicatively: ten steps at 95% per step is about 60% end to end. Reduce steps, add verification, or add checkpoints.
- Debuggability and testability drop as autonomy rises; plan tracing and replay from the start.

**Common mistakes**
- "Agent" as default architecture.
- No stop conditions.
- Framework-first thinking (choosing LangGraph/CrewAI/etc. before defining the problem).

**Follow-ups**
- Convince me a workflow is not enough for this task.
- How do you measure whether adding autonomy improved outcomes?
- What is the minimum viable agent you would ship first?

---

## Q3.2 Compare agent architectures and patterns: ReAct, plan-and-execute, reflection, tree search, orchestrator-workers, and supervisor/handoff.

**Interviewer is testing:** Pattern vocabulary tied to trade-offs.

**Expected answer / key points**
- **ReAct:** interleave reasoning and acting; flexible, adaptive, but can meander and is token-hungry.
- **Plan-and-execute:** plan first, then execute steps (possibly with a cheaper model), replan on failure; better cost control and inspectability, but brittle when the plan is wrong.
- **Reflection / evaluator-optimizer:** a critic reviews and the actor revises; improves quality where criteria are checkable; risk of sycophantic self-approval.
- **Tree/graph search (ToT, LATS-style):** explore alternatives with scoring/backtracking; higher cost; valuable for hard reasoning/code problems with verifiers.
- **Orchestrator-workers:** a lead decomposes and delegates to specialists, then synthesizes; supports parallelism.
- **Supervisor/router and handoffs:** a router picks a specialist; agents transfer control with context.
- **Blackboard/shared state:** agents read and write a shared structure; decoupled but needs concurrency control.

**Senior/Lead considerations**
- Choose based on: predictability of steps, verifiability of results, latency budget, parallelism opportunity, and blast radius of mistakes.
- Verifiable environments (tests, schema, execution) make reflection and search far more effective.
- Cheap-model workers with a stronger planner/verifier is a common cost pattern.

**Common mistakes**
- Reflection with no ground-truth signal (the model rubber-stamps itself).
- Multi-agent for problems a single agent with good tools solves.

**Follow-ups**
- Plan-and-execute: what triggers replanning?
- How do you prevent the critic and actor from sharing the same blind spot?
- Where does parallel tool execution help and where does it break ordering assumptions?

---

## Q3.3 Single-agent versus multi-agent: when does multi-agent actually help?

**Interviewer is testing:** Ability to justify complexity.

**Expected answer / key points**
- Multi-agent helps with: **context isolation** (each subagent gets a focused context window), **parallelism** on breadth-heavy tasks (research, large refactors), **specialization** (different tools, prompts, models), **separation of duties** (author versus reviewer), and security boundaries (least-privilege per agent).
- Costs: many more tokens (coordination overhead), harder debugging, error propagation, context loss at handoffs, non-determinism in coordination, latency from serialization, conflicting edits to shared state.
- Guidance: prefer a single agent with strong tools; go multi-agent for breadth-first, parallelizable, or context-heavy work; give subagents **clear task specs and output contracts** and have them return compact results, not transcripts.

**Senior/Lead considerations**
- Define communication protocols: schemas for handoffs, shared state model, conflict resolution, ownership of the final answer.
- Budgeting: cap agents, depth of delegation and total tokens; avoid recursive spawn explosions.
- Evaluate at both the system and per-agent level.

**Common mistakes**
- Agents "chatting" in natural language with no contract.
- Every role as an agent (a "manager", "engineer", "QA" persona zoo) with no measured benefit.
- No termination condition for agent-to-agent dialogue.

**Follow-ups**
- Two subagents produce conflicting results; who arbitrates and how?
- How do you pass context to a subagent without dumping the whole transcript?
- How do you trace and debug a five-agent run?

---

## Q3.4 Explain agent memory: what types, where stored, and what can go wrong.

**Interviewer is testing:** Systems view of state beyond the context window.

**Expected answer / key points**
- **Working memory:** current context (scratchpad, plan, recent observations). **Episodic:** past runs/interactions. **Semantic:** facts about the user/domain. **Procedural:** learned instructions, playbooks, skills.
- Storage: context window, files/scratch stores, key-value/DB for structured state, vector store for semantic recall, graph for relations.
- Write policy: explicit memory tools versus background extraction; include provenance, timestamps, confidence, and scope (user/team/global).
- Read policy: retrieve by relevance, recency and importance; budgeted injection.
- Failure modes: memory poisoning, stale or contradictory facts, privacy leakage across users, bloat, self-reinforcing errors.

**Senior/Lead considerations**
- Memory should be **inspectable, editable and deletable** by users and admins.
- Separate durable memory from per-task scratch state; scratch state should expire.
- Gate writes from untrusted sources (web, documents) to prevent persistent injection.

**Common mistakes**
- Writing every observation to long-term memory.
- Using memory to hold secrets.

**Follow-ups**
- How do you evaluate that memory improves outcomes?
- How do you resolve conflicting memories?
- What does "forget me" require across all stores?

---

## Q3.5 What is MCP (Model Context Protocol) and how do you integrate tools at enterprise scale?

**Interviewer is testing:** Current tooling standards plus platform/security design.

**Expected answer / key points**
- MCP is an open protocol standardizing how AI applications (clients/hosts) connect to external tools, resources and prompts via servers, over JSON-RPC with local (stdio) and remote (HTTP-based) transports and an authorization model built on OAuth.
- Primitives: **tools** (actions), **resources** (readable context), **prompts** (templates); plus client features such as sampling and elicitation. Capability negotiation on connect.
- Benefits: write an integration once, reuse across clients; decouples agent logic from tool implementations; enables a registry/catalog.
- Enterprise design: central **MCP registry** (approved servers, versions, owners), **MCP gateway/proxy** (authn/authz, rate limits, audit, policy, secrets injection), per-user delegated auth (not shared service accounts), tool allowlists per agent/role, schema/version pinning, observability, and sandboxed execution for local servers.
- Alternatives: direct function calling, OpenAPI-to-tool adapters, agent-to-agent protocols for inter-agent communication.

**Senior/Lead considerations**
- Security threats specific to tool ecosystems: **tool poisoning** (malicious instructions in tool descriptions), **rug pulls** (server changes behavior after approval), **confused deputy** (server uses its own privileges for a user's request), token passthrough, over-broad scopes, and untrusted servers reading context.
- Pin and review tool definitions; alert on description changes; treat tool outputs as untrusted data.
- Too many exposed tools bloat context and reduce selection accuracy; use tool search/dynamic loading and namespaces.

**Common mistakes**
- Installing community servers without review.
- One powerful shared credential for all users.
- Exposing every tool of a server to every agent.

**Follow-ups**
- How do you do per-user auth from an agent through an MCP server to a downstream API?
- How do you version and roll back MCP servers?
- How would you sandbox a local MCP server that needs filesystem access?

---

## Q3.6 Orchestration and state management: how do you build durable, resumable agent workflows?

**Interviewer is testing:** Distributed-systems rigor applied to agents.

**Expected answer / key points**
- Model the agent as a **state machine/graph**: nodes (LLM calls, tools, validators), edges (conditions), explicit typed state.
- **Persist state at each step** (checkpointing) to a durable store so runs survive crashes, deploys and long waits; support **resume, replay, time-travel debugging, and forking**.
- Use a **durable execution engine** (workflow engines such as Temporal or cloud equivalents) or a checkpointer in an agent framework; run steps as activities with retries, timeouts and heartbeats.
- Long-running tasks: async job model (submit, poll/webhook), event-driven wakeups (human approval, external callback), idempotency keys per step.
- Determinism: keep orchestration logic deterministic; isolate nondeterministic LLM calls as recorded activities so replays reuse results.
- Concurrency: fan-out/fan-in with limits; per-tenant concurrency controls; queues with priorities.

**Senior/Lead considerations**
- Decide **where state lives** (in-context versus external store) based on size, sensitivity and replay needs.
- Schema evolution of persisted state across deploys.
- Cost visibility per run and per step.

**Common mistakes**
- Holding agent state in process memory.
- Re-executing side-effecting tools on retry.
- No run IDs correlating logs, traces and state.

**Follow-ups**
- A worker dies between a tool call and checkpoint write. What happens?
- How do you deploy a new agent version while thousands of runs are in flight?
- How do you implement cancellation?

---

## Q3.7 Human-in-the-loop: how do you design approvals and escalation?

**Interviewer is testing:** Practical safety and UX design.

**Expected answer / key points**
- Classify actions by **risk and reversibility**: read-only (auto), reversible writes (auto with notify or undo), irreversible or high-impact (require approval), forbidden (blocked).
- Approval patterns: pre-action approval with a clear diff/preview, post-action review with rollback, sampling audits, dual control for critical actions, confidence-based escalation.
- Implementation: agent pauses with a persisted state and an approval request (who, what, why, evidence, expiry); approver identity and decision are audited; resume on decision; timeouts and delegation rules.
- UX: show the exact action and parameters, not a model's paraphrase; avoid approval fatigue by batching low-risk items and auto-approving policy-safe ones.
- Feedback loop: approvals/rejections become labeled data for evals and policy tuning.

**Senior/Lead considerations**
- The approval UI must display **ground-truth parameters** (what will actually execute), otherwise a compromised model can mislead the human.
- Track approval rate, latency and override reasons as product metrics.
- Regulated domains may require named accountability and retention of records.

**Common mistakes**
- Asking approval for everything, so humans rubber-stamp.
- Showing model-written summaries rather than actual tool arguments.

**Follow-ups**
- How do you prevent approval fatigue?
- What happens if the approver never responds?
- How would you learn which actions can safely become automatic?

---

## Q3.8 How do you make agents reliable and recover from failures?

**Interviewer is testing:** Production reliability mindset.

**Expected answer / key points**
- Failure taxonomy: tool errors and timeouts, malformed calls, wrong tool selection, hallucinated arguments, context overflow, goal drift, infinite loops, partial completion, downstream side effects, model/provider outage.
- Mitigations:
  - **Tool layer:** schema validation, timeouts, retries with exponential backoff and jitter, circuit breakers, idempotency keys, compensating actions (saga pattern), sandboxed execution.
  - **Agent layer:** error messages returned to the model in actionable form, bounded retries, alternative-strategy prompts, verification steps (tests, assertions, cross-checks), checkpoint and resume.
  - **System layer:** fallbacks (smaller deterministic path, human handoff), graceful partial results, feature flags/kill switches, canary and shadow releases.
- Measure: task success rate, steps per task, tool error rate, retry rate, cost per success, time to completion, escalation rate.

**Senior/Lead considerations**
- Design the agent to **fail safely and visibly**: explicit "cannot complete" states with reasons.
- Prefer verifiable steps (run the tests, validate the output) over asking the model if it is done.
- Postmortems: replay failed traces into the eval suite.

**Common mistakes**
- Swallowing tool errors so the model "guesses".
- Retrying non-idempotent operations blindly.
- No definition of "done".

**Follow-ups**
- A tool intermittently returns success but did nothing. How do you detect it?
- How do you implement compensation for a multi-step booking flow?
- What is your kill-switch strategy?

---

## Q3.9 How do you prevent infinite loops and uncontrolled tool execution?

**Interviewer is testing:** Safety controls that do not depend on the model behaving.

**Expected answer / key points**
- **Hard limits enforced outside the model:** max steps/iterations, max tool calls (total and per tool), max wall-clock time, max tokens and dollar budget per run, max recursion/delegation depth, max parallel calls.
- **Loop detection:** hash recent (tool, args) pairs and detect repeats or oscillation; detect no-progress (state unchanged across N steps); escalate or abort with a structured failure.
- **Progress checks:** require a measurable state change or a new subgoal per step; periodic "are we making progress?" evaluation by a separate monitor.
- **Tool governance:** allowlists per agent/task, rate limits per tool and per user, spend caps for paid APIs, approval for destructive or costly tools, dry-run modes, sandboxes with network and filesystem restrictions.
- **Circuit breakers** at the platform level (tenant-wide caps, anomaly detection on tool-call rates), and a **kill switch** to halt runs.
- Fail with a useful partial result and an explanation rather than looping silently.

**Senior/Lead considerations**
- Enforce limits in the orchestrator/gateway so a prompt-injected or confused model cannot bypass them.
- Budget tiers by task class; alert on anomalies (for example a sudden 10× tool-call rate).
- Cost caps protect against **denial-of-wallet** attacks.

**Common mistakes**
- Limits stated only in the system prompt.
- Only step limits, no cost or time limits.
- No dedupe of repeated identical calls.

**Follow-ups**
- How do you distinguish a legitimately long task from a loop?
- What should the agent return when it hits a limit?
- How would you detect an agent spawning subagents recursively?

---

## Q3.10 Agent evaluation and improvement loops

**Interviewer is testing:** Ability to measure non-deterministic, multi-step systems.

**Expected answer / key points**
- Levels: **outcome** (did the task succeed?), **trajectory** (were the steps sensible, efficient, safe?), **component** (tool selection accuracy, argument correctness, retrieval quality).
- Methods: task suites with programmatic graders (tests pass, DB state correct), rubric/LLM judges (calibrated), human review sampling, simulation environments and user simulators, replay of production traces, pass@k and pass^k (consistency across repeated runs).
- Track cost, latency and step counts alongside success.
- Regression gating on prompt, tool, model or framework changes.

**Senior/Lead considerations**
- Non-determinism requires multiple trials and confidence intervals; single-run comparisons mislead.
- Beware of judge bias and leakage between train/eval sets.
- Use failed production traces to grow the eval set continuously.

**Common mistakes**
- Only checking the final answer text.
- Eval sets that drift from production distribution.

**Follow-ups**
- How many trials do you need to detect a 3-point regression?
- How do you evaluate safety (agent did something it should not have)?
- How do you eval an agent whose tools have side effects?

---

## Q3.11 Design the tool interface for an agent (agent-computer interface).

**Interviewer is testing:** Practical craft that strongly affects agent success.

**Expected answer / key points**
- Tools should be **few, clear, well-named, non-overlapping**, with descriptions that explain when to use them, parameter semantics, examples and edge cases.
- Return **concise, relevant, model-friendly outputs** (paginate, filter, summarize; include IDs for follow-up); actionable error messages ("no match; try broader query X").
- Prefer higher-level task tools over thin API wrappers when it reduces steps; but keep primitives for flexibility (for example read/search/edit tools for code).
- Make destructive actions explicit and separate; require confirmation tokens where appropriate.
- Test tools with the agent and iterate on failure transcripts; poka-yoke design (make mistakes hard: absolute paths, enums, required fields).

**Senior/Lead considerations**
- Tool definitions are prompts; treat them as versioned, evaluated artifacts.
- Token efficiency of tool outputs strongly affects cost and success.
- Namespacing and dynamic tool loading scale to hundreds of tools.

**Common mistakes**
- Thin wrappers returning giant JSON blobs.
- Ambiguous parameter names, overlapping tools.

**Follow-ups**
- The agent keeps calling the wrong tool. How do you debug?
- How would you expose 500 internal APIs to an agent?

---

## Q3.12 Coding agents and computer-use agents: what is architecturally different?

**Interviewer is testing:** Understanding of agents in open-ended, high-risk environments.

**Expected answer / key points**
- Coding agents: repo understanding (search, symbol graph, retrieval), edit tools (patch/diff based), execution loop (run tests/linters/build) as the verifier, sandboxed containers, git branching, PR-based review, and long-horizon context management.
- Computer-use/browser agents: perceive UI (screenshots/DOM/accessibility tree), act via clicks/keystrokes; slower, more brittle, higher prompt-injection exposure from web content.
- Both need sandboxing, network egress control, credential isolation, and strong logging.
- Verifiers matter: tests, type checks, screenshots diffs, DOM assertions.

**Senior/Lead considerations**
- Ground the agent in **executable feedback**; success without a verifier is guesswork.
- Isolation per task (ephemeral VM/container), no long-lived credentials, restricted network.
- Diff-size and blast-radius limits; require human review for merges.

**Common mistakes**
- Giving an agent broad shell access to a developer workstation or production credentials.
- No test signal, so agents "fix" things by weakening tests.

**Follow-ups**
- How do you prevent the agent from gaming the tests?
- How do you handle a 1M-line monorepo within limited context?

---

# 4. Production Hosting & Deployment

## Q4.1 Managed APIs versus self-hosted open-weight models: how do you decide?

**Interviewer is testing:** Build-vs-buy reasoning with real constraints.

**Expected answer / key points**
- **Managed API strengths:** frontier quality, zero infra, fast iteration, elastic capacity, provider-side optimizations and safety, built-in features (caching, batch, structured outputs).
- **Managed API risks:** data governance/residency, rate limits and quota, vendor lock-in and deprecations, per-token pricing at scale, limited customization, outage dependency, latency variability.
- **Self-hosting strengths:** data control, customization (fine-tuning, adapters), predictable cost at high sustained utilization, latency control, no per-token pricing, offline/air-gapped operation.
- **Self-hosting costs:** GPU procurement/capacity, MLOps/SRE staffing, serving optimization, upgrade and security burden, lower peak quality unless a strong open model fits, idle capacity waste.
- Decision factors: quality requirement, volume and utilization (break-even analysis), compliance, latency, customization need, team capability, time to market.
- Common answer: **hybrid**: managed frontier models for hard/low-volume tasks, self-hosted small/medium models for high-volume/narrow tasks, behind an LLM gateway.

**Senior/Lead considerations**
- Do a break-even model: cost per million tokens at your actual utilization (peak-to-average ratio, redundancy, headroom) versus API price. Sub-50% utilization usually favors managed.
- Abstract providers behind a gateway to keep optionality.
- Consider provider private-networking, zero-retention agreements, and regional deployment before concluding self-hosting is needed for compliance.

**Common mistakes**
- Comparing GPU hourly price to API token price without utilization, redundancy and staffing.
- Assuming open models match frontier quality on your task without evals.

**Follow-ups**
- Walk through your break-even calculation.
- What would make you move a workload from API to self-hosted?
- How do you handle provider deprecation of a model your product depends on?

---

## Q4.2 Design GPU infrastructure and model serving for LLM inference.

**Interviewer is testing:** Practical knowledge of hardware, parallelism and serving stacks.

**Expected answer / key points**
- Sizing: weights + KV cache + activations/overhead must fit; choose GPU type by memory capacity, bandwidth, interconnect and precision support (for example FP8-capable parts).
- **Parallelism:** tensor parallel (split layers across GPUs; needs fast interconnect such as NVLink; use within a node), pipeline parallel (split layers across stages; across nodes; bubbles), data parallel/replicas (scale throughput), expert parallel (MoE).
- **Serving engines:** vLLM (PagedAttention, continuous batching), SGLang (RadixAttention prefix caching, structured decoding), TensorRT-LLM (NVIDIA-optimized, compile step), TGI, Triton as a serving layer, llama.cpp for CPU/edge. Compare on throughput, latency, model coverage, quantization support, operational maturity.
- Deployment: container image with pinned CUDA/driver/engine versions, model weights pulled from a registry/object store (cache on local NVMe, pre-warm), health/readiness checks tied to model load, GPU device plugin, node pools per GPU type, topology-aware scheduling.
- Cold start: model load can take minutes; mitigate with weight caching, fast storage, snapshotting, warm pools, and slower scale-in.

**Senior/Lead considerations**
- Replica sizing: choose the smallest tensor-parallel degree that fits the model with adequate KV headroom, since more replicas beat larger TP for throughput.
- Bin-packing small models with MIG/time-slicing or multi-model servers for utilization.
- Plan for GPU failures (ECC errors, XID faults, node loss) with health probes and automatic replacement.

**Common mistakes**
- Tensor parallelism across slow interconnect.
- Autoscaling GPUs like stateless CPU pods (slow start, expensive idle).
- Unpinned versions causing silent performance/behavior changes.

**Follow-ups**
- 70B model, 200 concurrent 4k-context users: how many GPUs and what layout?
- How do you reduce a 6-minute cold start?
- vLLM versus TensorRT-LLM: how do you choose?

---

## Q4.3 Kubernetes and containerization for AI workloads: what is different from typical services?

**Interviewer is testing:** Platform engineering depth.

**Expected answer / key points**
- GPU scheduling via device plugins and node labels/taints; separate node pools by GPU type; topology awareness for multi-GPU pods; requests/limits for GPU, CPU, memory and shared memory (`/dev/shm` sizing).
- Large images and model weights: image layering, lazy pulling, weights on shared/local volumes or fetched at startup with caching; init containers for warmup.
- Readiness must reflect **model loaded and warmed**, not just process up; graceful termination must drain in-flight streams (long-lived connections) with adequate `terminationGracePeriodSeconds`.
- Stateful aspects: KV cache is per-replica memory; routing can exploit prefix locality.
- Frameworks/tools: KServe, Ray Serve, KubeRay, Knative, Kueue/Volcano for batch/queue scheduling, Karpenter/cluster autoscaler for nodes, DCGM exporter for GPU metrics.
- Multi-tenancy: namespaces, quotas, network policies, pod security, per-tenant node pools for isolation when needed.

**Senior/Lead considerations**
- GPU nodes take minutes to provision; keep **buffer capacity** and prioritize with preemption for interactive over batch.
- Spot/preemptible GPUs suit batch and offline evaluation, not latency-critical serving without redundancy.
- Cost visibility per namespace/team through GPU-hour attribution.

**Common mistakes**
- HPA on CPU for GPU inference.
- Killing pods mid-stream during deploys.
- Ignoring shared memory and NCCL configuration for multi-GPU pods.

**Follow-ups**
- How do you roll out a new model version without dropping streams?
- How do you schedule a mix of interactive and batch inference on the same cluster?
- How do you isolate a noisy tenant?

---

## Q4.4 Autoscaling and load balancing for LLM inference

**Interviewer is testing:** Understanding that LLM load is not request-count-shaped.

**Expected answer / key points**
- Requests vary by orders of magnitude in cost (prompt and output length), so **RPS is a poor scaling signal**. Better: queue depth, in-flight sequences, **KV cache utilization**, tokens/s per replica, TTFT/ITL percentiles versus SLO, GPU utilization (with care).
- Custom-metric autoscaling (KEDA or HPA with custom metrics), scale-out on leading indicators, **slow scale-in** with drain, minimum warm replicas, and scheduled scaling for known patterns.
- **Load balancing:** least-outstanding-requests or cache-aware routing, not round-robin. **Prefix/session-affinity routing** improves KV cache hit rates; balance against hot spots.
- Admission control and load shedding: queue with bounds, prioritize by tier, reject early with `429`/`503` and `Retry-After`, degrade to smaller model.
- Separate pools for different classes (short interactive versus long-context or batch) to avoid head-of-line blocking.

**Senior/Lead considerations**
- Because scale-up takes minutes, overprovision headroom for burst or absorb bursts with queues and a fallback tier (managed API overflow).
- Define **SLO-based** scaling targets, not utilization targets.
- Test with realistic token-length distributions and bursty arrival patterns.

**Common mistakes**
- Round-robin across replicas with very different queue states.
- Scaling in aggressively, killing in-flight generations.
- Load tests with uniform prompts.

**Follow-ups**
- Traffic doubles in 30 seconds; what happens in your design?
- How does cache-aware routing interact with fairness?
- What metric would you alert on before users notice latency?

---

## Q4.5 High availability and disaster recovery for AI systems

**Interviewer is testing:** Classic reliability engineering applied to AI components.

**Expected answer / key points**
- Identify components and their failure domains: gateway, orchestrator, model serving pool, vector DB, feature/metadata stores, object storage, queues, third-party model APIs, identity provider.
- HA: multiple replicas across availability zones, no single points of failure, health-checked routing, graceful degradation (cached answers, smaller model, retrieval-only mode, "read-only" mode).
- **Multi-provider/model fallback** for external APIs, with prevalidated equivalents.
- DR: define **RPO/RTO** per component; vector index backups/snapshots or reproducible rebuild from source of truth; replicate model artifacts and configs across regions; infrastructure as code; regular game days and failover drills.
- Data durability: separate source-of-truth (documents, conversations, state) from derived stores (indexes, caches), which can be rebuilt.

**Senior/Lead considerations**
- Tier services: the LLM feature is often not the most critical path; degrade the feature, not the product.
- Rebuilding a large vector index may take hours to days; snapshots may be necessary to meet RTO.
- Test fallbacks for behavior differences (format, safety, quality), not just connectivity.

**Common mistakes**
- Assuming a managed API is "always up".
- Backups never restored in practice.
- Fallback model never evaluated.

**Follow-ups**
- Your primary model provider is down for 3 hours. Walk me through the user experience.
- How do you meet RTO of 30 minutes for the vector store?
- What is your degraded mode for retrieval outage?

---

## Q4.6 Multi-region deployment for AI applications

**Interviewer is testing:** Global architecture, data residency and cost awareness.

**Expected answer / key points**
- Drivers: latency (users near regions), availability, **data residency/sovereignty**, capacity (GPU scarcity is regional).
- Patterns: active-active with geo-routing (global load balancer/DNS/anycast); active-passive for cost; regional cells for isolation; per-region model pools and indexes.
- Data: region-pinned tenant data; replicated global reference data; **do not replicate regulated data across borders**; embeddings and logs also carry residency obligations.
- Model availability differs by region and provider; capacity quotas are per region; keep parity via IaC and config-driven model catalogs.
- Failover rules must respect residency (fail to an allowed region only); handle in-flight streams and state.
- Cross-region costs: egress, duplicated indexes and GPUs.

**Senior/Lead considerations**
- Cell-based architecture limits blast radius.
- Deployment: progressive rollout region by region with automated rollback.
- Observability aggregated globally but with residency-aware log handling.

**Common mistakes**
- Global vector index mixing regulated tenants.
- Failover to a region that violates residency.
- Assuming identical model versions/features in every region.

**Follow-ups**
- EU customers require processing in the EU only; what are the implications for logging, tracing and evaluation data?
- How do you keep model/prompt versions consistent across regions?

---

## Q4.7 API gateways versus LLM gateways: what belongs where?

**Interviewer is testing:** Clear separation of concerns.

**Expected answer / key points**
- **API gateway:** generic edge concerns: TLS termination, authN (OIDC/JWT/mTLS), coarse authZ, WAF, request size limits, request-based rate limiting, routing to services, DDoS protection.
- **LLM gateway:** LLM-specific: multi-provider abstraction, token-based quotas and budgets, model routing and fallbacks, prompt/response guardrails, semantic and prompt caching, usage/cost attribution, streaming-aware processing, model catalog, evaluation/shadow traffic hooks.
- They can be layered (API gateway in front, LLM gateway behind) or combined in one product; keep responsibilities clear to avoid duplicated policy.
- Streaming: support SSE/WebSocket/gRPC streaming; propagate cancellation (client disconnect must cancel upstream generation to save GPU).

**Senior/Lead considerations**
- **Client disconnect cancellation** materially reduces wasted cost.
- Idle timeouts on proxies and load balancers frequently cut long streams; tune them.
- Central policy but team-level autonomy: self-service onboarding with guardrails.

**Common mistakes**
- Load balancer timeouts killing long generations.
- Not propagating cancellation, so GPUs keep generating for departed users.

**Follow-ups**
- How do you meter tokens for streamed responses reliably?
- How would you allow teams to bring their own provider keys while keeping governance?

---

## Q4.8 Security for hosted AI: authentication, authorization and secrets

**Interviewer is testing:** Enterprise security fundamentals adapted to AI.

**Expected answer / key points**
- **AuthN:** OIDC/OAuth 2.x for users, workload identity (SPIFFE, cloud IAM roles) for services, mTLS in the mesh; short-lived tokens.
- **AuthZ:** RBAC/ABAC/ReBAC for model access, tools, data, prompts and admin operations; **on-behalf-of (delegated) tokens** so tool calls run with the end user's rights; least privilege per agent and per tool.
- **Secrets:** vault/KMS/secret manager, no secrets in prompts, images or repos; rotation; inject at runtime; the model should never see credentials.
- **Network:** private endpoints/VPC peering to model providers, egress allowlists, segmentation, WAF, private registries for images and weights.
- **Data:** encryption in transit and at rest, customer-managed keys where required, retention and deletion policies, log redaction.
- **Supply chain:** signed images, verified model weights (checksums/provenance), SBOMs, dependency scanning, safe serialization formats (avoid arbitrary pickle loading).
- Audit logging of access, prompts and tool actions with tamper resistance.

**Senior/Lead considerations**
- Threat-model the system (STRIDE plus LLM-specific threats).
- Separate duties for model registry, deployment, and data access.
- Prompts and traces contain sensitive data; they need the same protection as the source data.

**Common mistakes**
- A single service account with broad privileges behind the agent.
- Secrets placed in system prompts or tool descriptions.
- Downloading unverified model weights or using pickled checkpoints.

**Follow-ups**
- How do you propagate user identity from chat UI to the third-party API a tool calls?
- How would you detect a compromised model artifact?

---

## Q4.9 CI/CD, environments and release strategy for LLM applications

**Interviewer is testing:** MLOps/LLMOps maturity.

**Expected answer / key points**
- Versioned artifacts: code, prompts, tool schemas, model versions/adapters, retrieval config, index versions, guardrail policies, eval datasets.
- Pipeline: unit and integration tests, **offline eval gate** (quality, safety, latency, cost thresholds), security scans, staging with realistic data, canary release with automated rollback triggers, then progressive rollout.
- Shadow deployments: run new version on mirrored traffic, compare outputs and metrics before serving.
- Feature flags and kill switches for prompts, models and tools independent of code deploys.
- Rollback must revert **the whole configuration set** (prompt + model + index) atomically; index migrations are blue/green.
- Environment parity: same model versions and configs, sanitized data.

**Senior/Lead considerations**
- Treat model/provider upgrades as releases with eval gates; pin versions and schedule upgrades before deprecation dates.
- Coupling between embedding model and index requires coordinated rollout.
- Keep audit trails of what version answered each request.

**Common mistakes**
- Editing prompts in the console in production.
- Rolling back code but not the prompt or index.

**Follow-ups**
- How do you canary a change whose quality can only be judged after user feedback arrives days later?
- What does your rollback do to in-flight agent runs?

---

## Q4.10 Serving frameworks and deployment options: compare and justify.

**Interviewer is testing:** Awareness of ecosystem and trade-offs.

**Expected answer / key points**
- **vLLM:** broad model support, PagedAttention, continuous batching, prefix caching, wide community; strong default.
- **SGLang:** efficient prefix reuse (radix tree), structured output speed, good for agent/multi-call workloads.
- **TensorRT-LLM (+ Triton/Dynamo):** best-in-class NVIDIA performance with compilation and FP8/INT4 optimizations; more operational complexity.
- **TGI:** Hugging Face ecosystem integration.
- **Managed inference platforms:** cloud provider services (for example Bedrock, Vertex AI, Azure AI/OpenAI) and third-party inference hosts; trade control for convenience.
- **Edge/CPU:** llama.cpp/GGUF, ONNX Runtime for small models.
- Selection criteria: model support, hardware, throughput/latency at your workload, quantization formats, structured decoding, LoRA serving, observability hooks, maintainer health, and your team's operational skills.

**Senior/Lead considerations**
- Benchmark on **your** prompt/output distributions; published benchmarks use synthetic loads.
- Avoid deep lock-in: standardize on an OpenAI-compatible interface at the serving boundary.
- Track upstream release cadence; plan upgrade testing.

**Common mistakes**
- Choosing on a blog benchmark.
- Building a custom server unnecessarily.

**Follow-ups**
- Design a benchmark plan to choose between two engines.
- What metrics do you report (goodput at SLO, p50/p95/p99 TTFT and ITL, tokens/s/GPU, cost per million tokens)?

---

# 5. LLM / Agentic AI System Design

> **Format note:** For design questions, the "Expected answer" is an outline you should be able to draw on a whiteboard in about 35 minutes. Always begin with clarifying questions and capacity estimates, then walk the framework from the top of this document.

### Reusable back-of-envelope estimates (memorize the method)

```
Requests/day → peak QPS (× 3 to 10 for peak-to-average)
Tokens/request (input + output) → tokens/s at peak
Concurrency (Little's Law) = QPS × average end-to-end latency (s)
Per-GPU throughput (tokens/s at target SLO) → GPUs needed = peak tokens/s ÷ per-GPU tokens/s ÷ target utilization (0.5 to 0.7)
KV cache per sequence × concurrency ≤ GPU memory − weights − overhead
Cost/request = input tokens × in-price + output tokens × out-price (+ retrieval, rerank, guardrail, judge calls, retries)
```

---

## Q5.1 Design a production-grade RAG platform (multi-tenant, enterprise)

**Interviewer is testing:** End-to-end platform thinking: ingestion, retrieval quality, security, evaluation, operations.

**Expected answer / key points**
- **Requirements to clarify:** number of tenants, corpus size and growth, source systems, freshness SLA (minutes vs daily), latency SLO (for example TTFT < 1.5 s), answer quality bar, compliance/residency, ACL model, modalities.
- **Ingestion plane:** connectors (SharePoint, Drive, Confluence, DBs, tickets) → change-data-capture/webhooks + periodic reconciliation → queue → parsers (layout-aware, OCR, table extraction) → normalizer/dedupe → chunker → enrichment (metadata, ACLs, entity tags, optional contextual summaries) → embedding workers (batched, autoscaled) → index writers. Idempotent, retryable, with dead-letter queues and per-document status.
- **Storage:** object store (raw + parsed, source of truth), metadata DB (documents, versions, ACL snapshots), vector index plus keyword index (hybrid), separate namespaces/partitions per tenant or tier.
- **Serving plane:** API + auth → query understanding (rewrite, intent, filters) → hybrid retrieval with ACL pre-filter → rerank → context builder (token budget, dedupe, citations) → LLM via gateway → grounding/citation validation → streamed response.
- **Cross-cutting:** evaluation service (golden sets per tenant, nightly runs), observability (per-stage latency, recall proxies, feedback), cost controls, admin UI (source health, reindex, deletion), audit.
- **Scale estimate example:** 500M chunks × 1,024 dims × 4 B ≈ 2 TB raw vectors before compression; use quantization/PQ or sharding; HNSW memory overhead is significant, so shard by tenant and tier.
- **Failure modes:** parser failures on odd files, ACL sync lag, stale index, embedding-model migration, hot tenants, index rebuild time, prompt injection through documents.

**Senior/Lead considerations**
- Treat **freshness and deletion** as product requirements with SLAs; build the index as a derived store rebuildable from the source of truth.
- Tenant isolation model chosen per data sensitivity; **noisy neighbor** controls on ingestion and query paths.
- Provide per-tenant relevance tuning without forking the pipeline (config-driven).
- Sequence roadmap: v1 hybrid + rerank + ACL + eval harness; v2 freshness/CDC, multimodal; v3 agentic retrieval, per-tenant tuning.

**Common mistakes**
- Jumping to vector DB brand choice before requirements.
- No eval harness or per-stage metrics.
- Ignoring ACL propagation and deletion.

**Follow-ups**
- Index rebuild for a 1B-chunk corpus: how long, how do you cut over?
- A single tenant ingests 50 million documents in a day. What protects others?
- How do you attribute cost per tenant?
- How do you prove to an auditor that a deleted document is gone from every store (index, cache, backups, logs)?

---

## Q5.2 Design an enterprise AI assistant (chat + actions across company systems)

**Interviewer is testing:** Combining RAG, tools, identity, governance and UX at scale.

**Expected answer / key points**
- **Clarify:** user count, channels (web, Slack/Teams, mobile), systems to integrate, read-only versus actions, regulated data, latency, adoption metrics.
- **Architecture:** client apps → API gateway (SSO/OIDC) → conversation service (sessions, history, memory) → orchestrator (router → RAG path, tool/agent path, or direct chat) → LLM gateway → tools via **MCP gateway/tool registry** → downstream systems using **delegated user tokens**.
- **Knowledge:** federated retrieval across enterprise sources with ACL enforcement; source ranking and citations.
- **Actions:** typed tool schemas, read/write classification, approval flows for writes, audit trail; idempotency.
- **Governance:** policy engine (who can use which tools/models/data), DLP/PII controls, content policies, retention, admin analytics, kill switches.
- **Personalization/memory:** opt-in, transparent, deletable.
- **Quality:** eval sets by domain, feedback capture, online metrics (task completion, deflection, citation clicks, thumbs), human review queues.
- **Rollout:** pilot group → expansion; per-department enablement; training and change management.

**Senior/Lead considerations**
- The hard part is **integration + identity + trust**, not the chat UI.
- Start read-only and grow toward actions as reliability data accumulates.
- Cost governance: per-user/department budgets and model tiering.

**Common mistakes**
- A service account with access to everything to "simplify" integration.
- No admin visibility into what the assistant did.
- Launching to everyone without domain-level evals.

**Follow-ups**
- How does the assistant act "as" the user in a downstream system that has no OAuth support?
- HR data is sensitive: how do you scope access and logging?
- How do you measure ROI and decide where to invest next?

---

## Q5.3 Design a multi-agent workflow (for example, automated market research or incident response)

**Interviewer is testing:** Practical orchestration, contracts, cost control and failure handling.

**Expected answer / key points**
- **Clarify:** goal, latency (minutes vs hours), acceptable autonomy, data sources, output format, human checkpoints.
- **Example (research):** Orchestrator (plans, decomposes into sub-questions) → parallel **researcher subagents** (search/fetch/analyze; each returns a structured findings object with sources and confidence) → **verifier/critic** (fact-check against sources, detect gaps) → **synthesizer** (writes report with citations) → optional human review.
- **Contracts:** JSON schemas for task specs and results; shared state store (facts, sources, open questions); idempotent task IDs.
- **Control:** budgets (max subagents, tokens, time, tool calls), dedup of searches, depth limits, timeouts, partial results on timeout.
- **Runtime:** durable workflow engine/queue; workers autoscaled; state checkpoints; trace tree per run.
- **Quality:** citation verification, deduplication, contradiction detection; eval with rubric-based judges plus human sampling.
- **Failure handling:** failed subtask retried or reassigned; orchestrator degrades scope; final report states gaps explicitly.

**Senior/Lead considerations**
- Show that multi-agent is justified: breadth, context isolation, parallelism; quantify the token multiplier and value.
- Use cheaper models for workers and a stronger model for planning/synthesis/verification.
- Conflict resolution and ownership of final answer are explicit.

**Common mistakes**
- Agents conversing free-form.
- No budget or stop criteria.
- No verification of subagent claims.

**Follow-ups**
- Two researchers return contradictory facts; what does the system do?
- How do you keep cost per report predictable?
- How would you test this system deterministically?

---

## Q5.4 Design an AI coding agent (for example, autonomous bug-fix / PR agent)

**Interviewer is testing:** Agent loops with verifiers, sandboxes and safety.

**Expected answer / key points**
- **Flow:** issue/task intake → sandbox provisioning (ephemeral container with repo checkout at a commit, dependencies cached) → context building (repo map, code search, symbol/dependency graph, relevant files via retrieval, recent history) → plan → edit (patch/diff tools, not whole-file rewrites) → run **verifiers** (build, unit tests, linters, type checks, security scans) → iterate on failures → self-review → open PR with explanation, test evidence and risk notes → human review → merge via existing CI/CD.
- **Tools:** read/search, edit/patch, run command (sandboxed), test runner, git (branch, commit, push to a fork/branch only), issue tracker.
- **Safety:** no production credentials; network egress allowlist; resource limits; secrets scanning on diffs; diff-size limits; protected paths (CI config, auth, infra) requiring extra review; no direct pushes to main.
- **Scale:** job queue, sandbox pool with warm images, per-repo caches, concurrency limits, cost budgets per task.
- **Evaluation:** benchmark on internal historical issues (with tests as ground truth), pass rate, PR acceptance rate, reviewer effort, regression rate, cost per merged PR.
- **Failure modes:** flaky tests, test tampering, over-broad edits, context overflow in large repos, hallucinated APIs, dependency confusion/prompt injection from issue text or repo content.

**Senior/Lead considerations**
- The **verifier signal** (tests, CI) is the backbone; invest in fast, reliable environments.
- Guard against reward hacking (deleting tests, special-casing).
- Progressive autonomy: suggestions → draft PRs → auto-merge for narrowly scoped, high-confidence classes.

**Common mistakes**
- Running arbitrary commands on shared infrastructure.
- No test evidence in the PR.
- Ignoring that issue text is untrusted input.

**Follow-ups**
- How does the agent navigate a very large monorepo?
- How do you prevent the agent from modifying its own tests to pass?
- How do you decide when a task is too risky for autonomy?

---

## Q5.5 Design an LLM gateway supporting multiple models and providers

**Interviewer is testing:** Platform design, reliability and cost governance (extends Q2.7).

**Expected answer / key points**
- **API layer:** OpenAI-compatible surface + native passthrough; streaming; auth via API keys/JWT; tenant/team/project identity.
- **Core pipeline:** request validation → authN/Z → policy (allowed models, data classification) → rate limit/quota (RPM/TPM/budget) → guardrails (input) → cache lookup → router → provider adapter (translation of request/response, tokenizer/pricing metadata) → retry/fallback/circuit breaker → guardrails (output) → usage logging → response.
- **Control plane:** model catalog, routing policies, quotas/budgets, key management (provider keys in a secret store), prompt registry, evaluation/shadow configs; config distribution with validation and staged rollout.
- **Data plane scaling:** stateless proxy replicas; low overhead (single-digit ms); async usage/log pipeline via queue; distributed rate limiter (Redis/token-bucket with local approximations).
- **Reliability:** health checks per provider/region/model, outlier detection, hedging carefully (double cost), adaptive concurrency limits, bulkheads per tenant.
- **Observability:** trace per request, tokens in/out, cost, latency (TTFT/total), cache hit, fallback reason, error class.
- **Streaming metering:** count input up front; count output as streamed; reconcile with provider-reported usage; handle client disconnects (cancel upstream).

**Senior/Lead considerations**
- Fallback and routing policies encoded with **compliance constraints** (residency, allowed providers per data class).
- Capacity: manage provider quotas as a shared resource; priority classes when throttled.
- Governance: chargeback, budgets, anomaly alerts.

**Common mistakes**
- Global rate limiter as a single-point bottleneck.
- Silent fallback to models with different behavior.
- Logging full prompts without redaction.

**Follow-ups**
- Provider returns 429s for 10 minutes; how do you keep high-priority traffic flowing?
- How do you avoid a thundering herd on recovery?
- How do you handle providers with different tool-calling semantics?

---

## Q5.6 Design a scalable inference platform (self-hosted models for many teams)

**Interviewer is testing:** Infrastructure, multi-tenancy, scheduling, cost efficiency.

**Expected answer / key points**
- **Clarify:** models and sizes, number of teams, workload types (interactive, batch, fine-tuning), SLOs, GPU inventory, budget.
- **Layers:** model registry (versioned weights, quantized variants, adapters, provenance) → deployment controller (declarative model specs: model, engine, TP degree, min/max replicas, SLOs) → serving pools on Kubernetes GPU node pools → inference router (cache-aware, priority-aware) → gateway.
- **Scheduling:** interactive (reserved/guaranteed capacity), batch (preemptible, backfill), fine-tuning (queued); GPU sharing (MIG) for small models; bin-packing; fair-share quotas per team.
- **Autoscaling:** SLO- and queue-based, with warm pools; scale-to-zero only for non-critical models with cold-start tolerance.
- **Optimization services:** benchmarking pipeline, automatic quantization with eval gates, engine upgrades with canary.
- **Operations:** GPU health monitoring, automatic node replacement, capacity forecasting, cost attribution, model deprecation workflow.
- **Multi-tenancy:** namespaces, quotas, network policies, per-team keys, audit; optional dedicated pools for sensitive tenants.

**Senior/Lead considerations**
- **Utilization** is the economic driver; track goodput/GPU-hour, idle time, and fragmentation.
- Provide a paved road (standard deployment spec) to prevent bespoke snowflake deployments.
- Capacity planning includes lead times for GPU procurement or reservations.

**Common mistakes**
- Every team deploys its own dedicated cluster with 15% utilization.
- No path to retire models.

**Follow-ups**
- How do you price/charge back shared GPU usage fairly?
- A new frontier open model lands; what is your evaluation-to-production path and timeline?
- How do you handle a 405B-class model needing multi-node serving?

---

## Q5.7 Design an AI system handling millions of requests per day (or per hour)

**Interviewer is testing:** Capacity estimation, queueing, tiering, graceful degradation.

**Expected answer / key points**
- **Estimate:** for example 50M requests/day ≈ 580 avg QPS; peak ×5 ≈ 2,900 QPS. With 1,500 input + 300 output tokens per request: peak ≈ 4.35M input tokens/s (prefill) and 870k output tokens/s (decode) if all served synchronously by self-hosted models; that scale forces caching, routing to small models, and possibly managed capacity. (Show the arithmetic and use it to justify design choices.)
- **Tiering strategy:** (1) cache (exact/semantic/prompt) to eliminate repeats, (2) rules/classifiers to avoid LLM calls, (3) small fast model for the majority, (4) larger model for hard cases via cascade, (5) async/batch for non-interactive work.
- **Architecture:** global load balancing → regional cells → stateless API tier → queue with priority → inference pools + managed API overflow → results cache.
- **Protection:** rate limits per tenant, admission control, load shedding, backpressure, circuit breakers, bulkheads.
- **Data:** streaming logs to a lakehouse; sampled traces; cost/quality dashboards.
- **Reliability:** multi-AZ/region, degrade gracefully (shorter outputs, smaller model, cached answers).

**Senior/Lead considerations**
- At this scale, **1% efficiency gains are real money**; invest in routing, caching and distillation.
- Design for cost per request targets from the start.
- Define which requests can be delayed, and use that flexibility to smooth peaks.

**Common mistakes**
- Skipping arithmetic.
- One model for everything.
- No plan for a provider outage or GPU shortage.

**Follow-ups**
- Cache hit rate needed to meet budget?
- What is the failure behavior at 2× peak?
- How do you prioritize paying customers over free users during overload?

---

## Q5.8 Design an agent with reliable tool execution

**Interviewer is testing:** Reliability engineering for the action layer (the "tool runtime").

**Expected answer / key points**
- **Tool runtime as a service:** schema registry, argument validation, authorization (user-scoped), execution sandbox, timeouts, retries with backoff and jitter, circuit breakers, rate limits, result normalization/truncation, structured error taxonomy returned to the model.
- **Idempotency:** every side-effecting call carries an idempotency key derived from (run_id, step_id, tool, args hash); tools/downstream APIs dedupe. Distinguish **read**, **idempotent write**, **non-idempotent write**, and **destructive** classes with different policies.
- **Transactions across steps:** saga pattern with compensating actions, or two-phase "prepare/commit" tools (create draft → approve → commit).
- **State and recovery:** persist tool-call intent and results (outbox pattern), so crash recovery can reconcile "did it happen?" by querying the downstream system rather than re-issuing.
- **Verification:** post-conditions checked by deterministic code (record exists with expected fields), not by asking the model.
- **Observability:** span per tool call with args (redacted), latency, outcome, retries; alerts on error-rate anomalies.
- **Safety:** allowlists, approval gates, budgets, dry-run mode.

**Senior/Lead considerations**
- Push reliability into deterministic infrastructure; the model chooses, the runtime guarantees.
- Define exactly-once **effects** via idempotency, not exactly-once delivery.
- Separate credentials per tool with minimum scope.

**Common mistakes**
- Retrying non-idempotent writes.
- Trusting the model's statement that the action succeeded.
- Returning raw stack traces to the model.

**Follow-ups**
- The payment tool timed out; did it charge the customer? How does the agent proceed?
- How do you test tool failure handling systematically? (Fault injection.)
- How do you version tool schemas without breaking in-flight runs?

---

## Q5.9 Design an enterprise MCP platform

**Interviewer is testing:** Extensible, secure tool ecosystem design.

**Expected answer / key points**
- **Goals:** let teams publish tools once and let many agents/clients consume them safely; central governance without bottlenecking teams.
- **Components:**
  - **Registry/catalog:** server metadata, owners, versions, tool schemas, risk classification, data classes touched, SLAs, review status.
  - **MCP gateway/proxy:** single entry for remote servers; authN (SSO), **per-user delegated authZ** (token exchange/on-behalf-of), tool-level policy (allow/deny by role, agent, environment), rate limits/quotas, input/output filtering, audit logging, secret injection, response size limits, schema pinning and drift detection.
  - **Hosting:** managed runtime for internal servers (containers, sandboxed, egress-controlled), templates/SDKs, CI checks (schema lint, security scan, tests), staged promotion (dev → staging → prod).
  - **Client/agent integration:** discovery, dynamic tool loading (tool search) to avoid context bloat, namespacing, per-agent allowlists.
  - **Security review pipeline:** for third-party servers: provenance, static/dynamic analysis, permission review, description scanning for injection patterns; quarantine and pin versions.
  - **Observability:** per-tool usage, latency, error rate, cost, anomaly detection; lineage from agent run to tool call to downstream API.
- **Lifecycle:** versioning, deprecation, breaking-change policy, rollback.
- **Local versus remote:** local stdio servers on developer machines are high-risk; provide sanctioned, sandboxed alternatives.

**Senior/Lead considerations**
- Identity is central: never let a server act with its own broad privileges on behalf of a user (confused deputy).
- Pin tool definitions and alert on changes to descriptions (rug-pull/tool-poisoning defense).
- Balance governance with velocity: paved-road templates, automated approvals for low-risk tools.

**Common mistakes**
- Treating MCP servers as trusted because they are "internal".
- Exposing hundreds of tools directly in every agent's context.
- No lifecycle/ownership model.

**Follow-ups**
- A tool description changes overnight to include hidden instructions; how does the platform detect and respond?
- How do you support write-tools requiring human approval through MCP?
- How do you meter and charge back tool usage?

---

## Q5.10 Design a conversational customer-support agent with escalation

**Interviewer is testing:** Business-facing agent design with safety, metrics and handoff.

**Expected answer / key points**
- **Flow:** intent detection → retrieval over KB/policies/order data → answer or action (refund, reschedule) with limits → escalation to human with a full context summary.
- **Controls:** refund caps and policy checks in code; identity verification before account actions; PII redaction in logs; tone/brand guardrails; multilingual support.
- **Escalation triggers:** low confidence, sentiment, repeated failure, policy-restricted topics, explicit request, high-value customers.
- **Metrics:** containment/resolution rate, CSAT, escalation quality, average handle time, hallucination/policy-violation rate, cost per resolution.
- **Learning loop:** agent-assist mode first, then autonomy; feedback from human agents into KB and evals.

**Senior/Lead considerations**
- Optimize **resolution quality**, not containment alone (containment can hide bad experiences).
- Make policy decisions deterministic code; the model explains, not decides, monetary outcomes.

**Common mistakes**
- Letting the model decide refund amounts.
- No graceful handoff (customer repeats the story).

**Follow-ups**
- How do you handle an adversarial customer trying to social-engineer a refund?
- How do you evaluate before launch without real customers?

---

# 6. Scalability, Performance & Cost

## Q6.1 Define throughput, concurrency, TTFT, ITL and goodput. How do they interact and how do you set SLOs?

**Interviewer is testing:** Metric fluency and the ability to reason about queueing.

**Expected answer / key points**
- **TTFT:** time to first token = queueing + prefill (+ network + retrieval/guardrails upstream). **ITL/TPOT:** time between output tokens (decode speed as perceived by the reader). **End-to-end latency** ≈ TTFT + (output tokens − 1) × ITL. **Throughput:** tokens/s (or requests/s) per replica or cluster. **Concurrency:** in-flight requests (Little's Law: L = λ × W). **Goodput:** throughput of requests that **meet the SLO**.
- Interactions: raising batch size increases throughput but raises ITL and possibly TTFT via queueing; long prompts cause prefill spikes that stall decode for other users unless chunked prefill is used.
- SLOs: express as percentiles (for example p95 TTFT < 1.5 s, p95 ITL < 50 ms), per request class (short, long-context, batch). Human reading speed is roughly 5 to 8 tokens/s, so ITL targets are usually far below that unless output is consumed programmatically.
- Programmatic consumers care about **total latency**, not streaming smoothness.

**Senior/Lead considerations**
- Measure at the **client** (including network, gateway, retrieval), not only at the engine.
- Tail latency is driven by queueing and long-prompt interference; manage with admission control and pool separation.
- Choose the operating point on the throughput-latency curve deliberately; capacity planning uses goodput at the SLO, not peak throughput.

**Common mistakes**
- Reporting averages.
- Comparing engines at different SLO points.
- Ignoring upstream time (retrieval, tool calls, guardrails) in TTFT.

**Follow-ups**
- You have 300 concurrent users and average latency of 8 s; what QPS are you serving? (About 37.5 by Little's Law.)
- What is the effect of a 20k-token prompt arriving in the middle of a busy decode batch?

---

## Q6.2 Explain continuous batching and why it matters. What are its limits?

**Interviewer is testing:** Core serving-engine knowledge.

**Expected answer / key points**
- **Static batching** waits for a batch to complete; short sequences idle while long ones finish. **Continuous (iteration-level) batching** schedules each decode step, inserting new requests and removing finished ones immediately, so the GPU stays fully utilized and queueing drops.
- Enabled by PagedAttention-style memory management (no need for contiguous preallocated KV per max length).
- Scheduling policies: FCFS, priority, fairness, preemption (swap or recompute KV), max-batch-tokens and max-sequences limits.
- Limits: KV memory capacity, per-step latency growth with batch size, prefill interference (mitigated by chunked prefill), fairness issues, and head-of-line blocking from very long requests.

**Senior/Lead considerations**
- Tune `max_num_seqs`, `max_batched_tokens` and chunk sizes against your SLO, using load tests with realistic length distributions.
- Preemption behavior under memory pressure (recompute vs swap) affects tail latency.

**Common mistakes**
- Believing bigger batch is always better.
- Testing with fixed-length synthetic prompts.

**Follow-ups**
- What happens when KV memory runs out mid-generation?
- How do you protect short interactive requests from long batch jobs on the same engine?

---

## Q6.3 KV cache and prompt caching: how do they reduce cost and latency, and how do you design for hit rate?

**Interviewer is testing:** Practical cache engineering at the serving and API level.

**Expected answer / key points**
- **Engine-level prefix caching:** identical token prefixes reuse computed KV blocks (hash blocks or radix tree), skipping prefill; the biggest wins come from long shared system prompts, few-shot examples, tool definitions and shared documents.
- **Provider prompt caching:** cached input tokens are billed at a discount and are faster; requires exact prefix match, minimum lengths, TTL/refresh behavior and sometimes explicit cache markers (varies by provider; verify current rules).
- **Design for hits:** static content first, dynamic content last; deterministic serialization of tool definitions and few-shots; avoid timestamps/UUIDs early in the prompt; stable ordering; keep the prefix consistent across users where safe.
- **Routing for hits:** cache-aware or session-affinity routing to the replica that holds the prefix.
- Limits: cache eviction under memory pressure, cross-tenant isolation (do not share caches across tenants where prefix content is sensitive; timing side channels exist), TTLs.

**Senior/Lead considerations**
- Track **cache hit ratio, cached-token share, TTFT with/without hit**; regress-test prompt refactors for cache impact.
- In agent loops, the growing conversation is a prefix, so append-only context design gives high hit rates; editing earlier turns destroys them.
- Cache isolation: tenant-scoped or salted prefixes when required.

**Common mistakes**
- Injecting the current time at the top of the system prompt.
- Reordering tool definitions between calls.
- Assuming a cache hit means identical output cost (output tokens are still billed).

**Follow-ups**
- Cache hit rate dropped from 80% to 10% after a release; how do you investigate?
- What are the security implications of shared prefix caches?

---

## Q6.4 Streaming: how do you implement it end to end, and what breaks?

**Interviewer is testing:** Real-time systems details around LLMs.

**Expected answer / key points**
- Transport: SSE (simple, one-way, proxy-friendly), WebSockets (bidirectional), gRPC streaming (service-to-service). Chunked token deltas, with heartbeats to avoid idle timeouts.
- Benefits: perceived latency, progressive rendering, early cancellation.
- Challenges: guardrails and validation on partial output (buffer sentences/windows, or stream then retract), structured output parsing incrementally, tool-call streaming (partial JSON), proxy/LB buffering and timeouts, retries mid-stream (cannot cleanly retry after partial delivery; resume tokens or restart with dedupe), token metering for billing, backpressure, and mobile network drops.
- **Cancellation:** propagate client disconnect through gateway to engine to stop generation and free KV.

**Senior/Lead considerations**
- Decide a moderation strategy: pre-check input, post-check chunks with small delay, final full-response check for audit.
- Idempotent resume or "continue" semantics for long generations.
- Test through the actual proxy chain (CDN, WAF, ingress).

**Common mistakes**
- Response buffering by an intermediary, killing streaming.
- Not canceling upstream, burning GPU on abandoned requests.

**Follow-ups**
- How do you enforce a safety filter on streamed content?
- How would you resume a stream after a mobile disconnect?

---

## Q6.5 GPU utilization: how do you measure it, and why can "high utilization" be misleading?

**Interviewer is testing:** Ability to look beyond dashboard metrics.

**Expected answer / key points**
- `nvidia-smi` "GPU utilization" measures the fraction of time any kernel is running, not how efficiently the SMs or memory bandwidth are used. Better: **SM activity/occupancy, tensor-core utilization, memory bandwidth utilization, MFU (model FLOPs utilization) and MBU (memory bandwidth utilization)** via DCGM/profilers.
- Decode is bandwidth-bound: low tensor-core utilization is expected, so judge with MBU and goodput.
- Utilization killers: small batches, CPU-side overhead (tokenization, scheduling, Python), data-loading stalls, communication overhead in TP, imbalanced MoE experts, fragmentation, idle time from slow autoscaling, long-tail requests.
- Improve: larger effective batch via continuous batching, CUDA graphs, quantization, kernel fusion, better routing, pool consolidation, spot/batch workloads to fill valleys.

**Senior/Lead considerations**
- Tie utilization to cost: idle-capacity dollars and fragmentation are usually the largest waste.
- Combine interactive and batch workloads with priorities to smooth utilization.

**Common mistakes**
- Celebrating 95% `nvidia-smi` utilization.
- Ignoring the CPU/host as a bottleneck.

**Follow-ups**
- GPU shows 40% utilization but users see latency spikes; what do you check?
- How would you compute MBU for a decode-heavy workload?

---

## Q6.6 Rate limiting for LLM systems: how do you design it?

**Interviewer is testing:** Fairness, cost protection and distributed-systems detail.

**Expected answer / key points**
- Limit by **tokens** as well as requests (RPM and TPM), per user/tenant/API key/model, plus **concurrency limits** and **budget caps** (daily/monthly spend).
- Algorithms: token bucket/leaky bucket, sliding window; distributed enforcement via Redis or a sharded counter with local caching and periodic sync (accept small overshoot).
- **Streaming complication:** output tokens unknown up front; reserve based on `max_tokens`, then reconcile actual usage and refund the difference.
- Priority classes and weighted fair queuing; burst allowances; per-model limits; separate limits for expensive features (long context, tools).
- Client behavior: return `429` with `Retry-After` and rate-limit headers; encourage exponential backoff with jitter; server-side queues for smoothing.
- Upstream: respect provider limits by shaping outbound traffic (shared quota across tenants requires internal fairness).

**Senior/Lead considerations**
- Protect against **denial-of-wallet**: budgets and anomaly detection, not just RPS caps.
- Fair-share so a single tenant cannot exhaust the provider quota.
- Communicate limits transparently to customers.

**Common mistakes**
- Request-count-only limits.
- Consistency-strict global counters in the hot path.

**Follow-ups**
- How do you reserve and refund tokens for a streaming request?
- How do you handle internal batch jobs competing with interactive traffic for the same provider quota?

---

## Q6.7 Cost optimization: give me a prioritized plan to cut LLM spend by 40% without hurting quality.

**Interviewer is testing:** Structured, measurement-first optimization.

**Expected answer / key points**
1. **Measure:** cost per feature/tenant/request; break down by input/output tokens, model, retries, cache hits, judge/guardrail calls, embeddings, rerank, infra.
2. **Eliminate waste:** duplicate calls, runaway loops, retries, oversized `max_tokens`, unused features, verbose outputs, unnecessary judge calls.
3. **Prompt and context diet:** trim system prompts and few-shot, fewer/better retrieved chunks (rerank), compress history, drop unused tool definitions.
4. **Caching:** prompt caching (prefix design), response/semantic caching where safe, retrieval/embedding caching.
5. **Model right-sizing:** route/cascade, smaller models for easy tasks, distill hot paths, structured outputs to reduce verbosity.
6. **Batch and async:** discounted batch APIs and off-peak capacity for non-interactive work.
7. **Self-hosting economics:** quantization, higher utilization, consolidation, spot capacity for batch, reserved capacity for baseline.
8. **Governance:** budgets, alerts, chargeback, cost in eval reports.
- Validate each step against the eval suite; report **quality-adjusted** savings.

**Senior/Lead considerations**
- Attack the **top 2 or 3 cost drivers** first; the Pareto principle applies strongly.
- Set guardrails so savings do not degrade critical, high-risk flows.

**Common mistakes**
- Switching to a cheaper model globally without evals.
- Ignoring hidden multipliers (agent steps, retries).

**Follow-ups**
- Which step gives the fastest savings, and which gives the largest?
- How do you communicate trade-offs to product leadership?

---

## Q6.8 Capacity planning for an LLM service

**Interviewer is testing:** Quantitative planning skills.

**Expected answer / key points**
- Inputs: traffic forecast (daily/weekly seasonality, growth, launches), token distributions (p50/p95/p99 input and output), SLOs, per-GPU goodput at SLO (measured via load tests, not vendor specs), KV cache constraints, redundancy needs (N+1/N+2, AZ loss), headroom (typically 30 to 50%), scale-up time.
- Method: peak tokens/s ÷ per-replica goodput → replicas; multiply for redundancy; convert to GPUs and cost; add buffer for deploys and failures.
- Lead times: GPU procurement/reservation (weeks to months); commit-based discounts require forecasting; use managed-API overflow for uncertainty.
- Review cadence: monthly forecast versus actual, drift alarms, launch readiness checks.

**Senior/Lead considerations**
- Load-test with production-like distributions and burst patterns before launches.
- Blend reserved (baseline), on-demand (burst), spot (batch).
- Plan for model changes (a larger new model may double GPU needs).

**Common mistakes**
- Using average traffic and average lengths.
- No headroom for AZ failure.

**Follow-ups**
- Marketing launches a feature that may 5× traffic in a week; what do you do?
- How do you decide reserved versus on-demand mix?

---

## Q6.9 Batch and offline workloads (embedding, evaluation, backfills, summarization at scale)

**Interviewer is testing:** Throughput-oriented pipeline design.

**Expected answer / key points**
- Use **batch APIs** (typically discounted, asynchronous) or self-hosted engines run at max-throughput settings (large batches, no latency SLO).
- Pipeline: partitioned input in object storage → job queue/orchestrator (Airflow/Dagster/Ray/Spark) → workers with rate-aware clients → checkpointed outputs → validation → load to sink; **idempotent, resumable** by shard.
- Handle: rate limits and retries, poison inputs, partial failures, dedupe, deterministic sharding, cost tracking, and prioritization so batch does not starve interactive traffic.
- Embedding backfills: batch sizes, dimension/model versioning, dual-write during migration.

**Senior/Lead considerations**
- Schedule off-peak; preemptible capacity; separate quotas.
- Sample-based quality checks on batch outputs before committing to a full run (a bad prompt on 100M rows is expensive).

**Common mistakes**
- No checkpointing; a failure restarts everything.
- Running the full backfill before validating on a sample.

**Follow-ups**
- Summarize 200M documents in a week: what is your plan and budget?
- How do you detect silent quality regression mid-run?

---

# 7. LLM / Agent Evaluation & Observability

## Q7.1 Offline versus online evaluation: what does each give you, and how do you build the loop?

**Interviewer is testing:** Evaluation strategy maturity.

**Expected answer / key points**
- **Offline:** curated datasets (golden sets, edge cases, adversarial, regression cases from past incidents), automated metrics, LLM judges, human review; fast, repeatable, safe pre-release gates; risk of distribution mismatch.
- **Online:** real traffic: A/B tests, canary, shadow evaluation, implicit signals (edits, retries, abandonment, escalations, citation clicks), explicit feedback (thumbs), sampled human review, LLM-judge sampling on production traces; captures real distribution and drift; slower, riskier.
- Loop: production traces → sample/cluster failures → label → add to eval set → fix → offline gate → canary → monitor. Keep **train/dev/test separation** for prompt and fine-tune tuning to avoid overfitting to the eval set.
- Metrics hierarchy: business outcome → task success → component metrics → cost/latency/safety.

**Senior/Lead considerations**
- Version eval datasets; track eval-set coverage against production intent distribution.
- Statistical rigor: sample sizes, confidence intervals, multiple comparisons.
- Ownership: who maintains the golden set and judges.

**Common mistakes**
- Static eval set that never grows.
- Tuning prompts directly on the test set.
- Reporting a single aggregate score that hides regressions in critical segments.

**Follow-ups**
- Offline metrics improved, online metrics did not; why?
- How large must the eval set be to detect a 2-point change?

---

## Q7.2 How do you evaluate RAG systems?

**Interviewer is testing:** Component-level diagnosis.

**Expected answer / key points**
- **Retrieval:** recall@k, precision@k, MRR, nDCG against labeled relevant chunks/docs; context recall/precision.
- **Generation:** **faithfulness/groundedness** (claims supported by retrieved context), **answer relevance** (addresses the question), **completeness/correctness** against reference answers, citation accuracy, refusal/abstention appropriateness when evidence is missing.
- **End-to-end:** task success, user satisfaction, latency, cost.
- Dataset creation: real queries from logs (anonymized), SME-labeled, synthetic question generation from documents (with quality filtering), hard negatives, multi-hop and no-answer cases.
- Tooling: frameworks such as RAGAS, TruLens, DeepEval or custom harnesses; judge calibration against human labels.
- Diagnostic matrix: retrieval failed vs generation failed vs both, to target fixes.

**Senior/Lead considerations**
- Evaluate **per segment** (query type, tenant, document type, language).
- Track "unanswerable" behavior; hallucinating on missing evidence is the costly failure.
- Freshness and ACL correctness tests are part of RAG eval.

**Common mistakes**
- Only judging final answers with an LLM judge.
- Synthetic datasets that mirror the retriever's biases (questions generated from the same chunks).

**Follow-ups**
- How do you get ground truth labels cheaply and reliably?
- How do you validate an LLM judge? (Agreement with humans, position/verbosity bias tests, adversarial checks.)

---

## Q7.3 How do you evaluate agents (trajectories, tool use, safety)?

**Interviewer is testing:** Extension of evaluation to multi-step, side-effecting systems (see also Q3.10).

**Expected answer / key points**
- **Environments:** sandboxed simulators with mock or ephemeral versions of tools/databases so runs are safe and repeatable; state-based graders check end state (DB rows, files, tests).
- **Metrics:** success rate, pass^k (reliability over repeated trials), steps/tool calls, wasted calls, recovery from injected faults, cost and latency per success, escalation appropriateness, policy/safety violations.
- **Trajectory analysis:** judge intermediate reasoning and tool choices via rubrics; detect loops and unnecessary actions.
- **Adversarial suites:** prompt injection in tool outputs, tool failures, ambiguous instructions, conflicting goals.
- **Production:** trace sampling, anomaly detection, human review of high-risk actions.

**Senior/Lead considerations**
- Fault injection (timeouts, malformed responses) is essential to measure resilience.
- Combine deterministic graders with judges; keep judges for what code cannot check.

**Common mistakes**
- One run per task.
- Grading only the final message.

**Follow-ups**
- How do you make agent evals cheap enough to run on every PR?
- How do you keep the eval environment faithful as tools evolve?

---

## Q7.4 Hallucination detection, groundedness and faithfulness in production

**Interviewer is testing:** Applied verification techniques and their costs.

**Expected answer / key points**
- **Groundedness/faithfulness check:** decompose the answer into atomic claims; verify each against retrieved evidence using NLI models or LLM judges; compute a support score; flag or block unsupported claims.
- Other signals: **self-consistency** across samples, token-level uncertainty/logprobs (weak but useful), citation validation (does the cited span support the claim?), cross-model agreement, retrieval-confidence thresholds, and abstain-if-no-evidence rules.
- Deployment: inline for high-risk flows (adds latency), asynchronous sampling for monitoring; thresholds calibrated on labeled data to trade precision and recall.
- Actions on detection: revise with a corrective pass, show uncertainty, escalate to human, or refuse.

**Senior/Lead considerations**
- Verification cost can rival generation cost; use small specialized verifiers and risk-based triggering.
- Judge errors compound; measure the verifier's own precision/recall.

**Common mistakes**
- Treating logprobs as calibrated truth.
- Using the same model and prompt to verify its own output as the sole check.

**Follow-ups**
- Your verifier flags 15% of answers; support says most are false alarms. What do you do?
- How do you evaluate the verifier?

---

## Q7.5 Tracing, token/cost monitoring and latency monitoring: what do you instrument?

**Interviewer is testing:** Production observability design for AI systems.

**Expected answer / key points**
- **Tracing:** end-to-end distributed traces (OpenTelemetry-compatible; GenAI semantic conventions where available) with spans for retrieval, rerank, each LLM call, each tool call, guardrails, and agent steps; attributes: model, version, prompt version, tokens in/out, cached tokens, latency, finish reason, cost, tenant/user hash, error class. Correlate with a run/conversation ID.
- **Metrics:** request rate, error rate by class, TTFT/ITL/total latency percentiles, tokens/s, queue depth, cache hit rates, cost per request/tenant/feature, tool success rate, agent steps distribution, guardrail block rates, feedback rates.
- **Logs:** full prompts/responses stored under access control with **PII redaction, retention limits and sampling**; ability to replay.
- **Dashboards and alerts:** SLO burn rates, cost anomalies, quality-signal drops (judge scores, thumbs-down), provider error spikes, drift in input length distribution.
- Tools: LLM observability platforms (Langfuse, LangSmith, Arize, Helicone, etc.) or in-house on OpenTelemetry + warehouse.

**Senior/Lead considerations**
- Prompts and outputs are sensitive data; observability must follow data governance.
- Attribute cost to features/tenants for accountability.
- Cardinality and storage costs: sample intelligently (keep all errors and slow traces).

**Common mistakes**
- Logging everything indefinitely, including PII.
- No prompt version in traces, so regressions are untraceable.

**Follow-ups**
- How do you debug "the agent gave a wrong answer yesterday at 3 pm" for a specific user?
- How do you alert on quality degradation with no ground truth?

---

## Q7.6 Production feedback loops and regression testing for LLM applications

**Interviewer is testing:** Continuous improvement discipline.

**Expected answer / key points**
- Feedback capture: explicit (thumbs, corrections), implicit (copy, edit distance, retries, abandonment, follow-up questions, escalation), reviewer annotations.
- Triage: cluster failures (embedding-based topic clustering), prioritize by frequency × severity, root-cause by stage (retrieval, prompt, tool, model).
- Convert incidents into **regression tests** automatically; maintain tiered suites: smoke (minutes, every PR), full (hourly/nightly), deep/adversarial (pre-release).
- Gate changes on non-regression of critical slices; use paired comparisons on the same inputs; judge-based comparison with position-swapping to reduce bias.
- Model upgrades: run the full suite and shadow traffic; document behavior diffs; staged rollout.
- Guard against feedback bias (only unhappy or extreme users respond) and feedback loops that reinforce existing behavior.

**Senior/Lead considerations**
- Budget eval compute as part of the platform; cache judge results where prompts and outputs are unchanged.
- Governance for using user data in training/evals (consent, privacy, retention).

**Common mistakes**
- Thumbs-up rate as the only quality metric.
- Adding tests but never pruning or updating stale expectations.

**Follow-ups**
- Provider ships a model update; how do you know within a day if quality regressed?
- How do you handle flaky LLM-judged tests in CI?

---

# 8. Security & Responsible AI

## Q8.1 Prompt injection: direct versus indirect. How do you defend an agent that reads untrusted content and can act?

**Interviewer is testing:** Threat modeling and architectural (not prompt-only) defenses.

**Expected answer / key points**
- **Direct:** the user attempts to override instructions. **Indirect:** malicious instructions embedded in retrieved documents, web pages, emails, tool outputs, or images that the model ingests.
- No complete prompt-level fix; models cannot reliably distinguish instructions from data. Defend **architecturally**:
  - **Least privilege** for tools and credentials; scope per task and per user.
  - **Break the "lethal trifecta"**: avoid combining (1) access to private data, (2) exposure to untrusted content, and (3) ability to communicate externally/exfiltrate in one agent context.
  - Separate contexts/privilege levels: a quarantined model processes untrusted content and returns constrained, schema-validated output to a privileged planner (dual-LLM or capability-based designs).
  - **Human approval** for sensitive actions, showing actual parameters.
  - **Output controls:** block rendering of arbitrary markdown images/links (exfiltration via URL), allowlist egress domains, strip active content.
  - Provenance/taint tracking: mark data as untrusted and restrict what tainted data may influence.
  - Detection layers (classifiers, canary tokens) as defense-in-depth, never the sole control.
  - Monitoring and rate limits to detect anomalous tool patterns.

**Senior/Lead considerations**
- Assume compromise of the model's reasoning; design so that a fully compromised agent still cannot cause unacceptable harm (blast-radius limiting).
- Red-team continuously; keep a corpus of injection attacks in regression tests.

**Common mistakes**
- "We told the model to ignore malicious instructions."
- Relying on an injection classifier as the only defense.
- Allowing arbitrary outbound URL fetches or markdown image rendering.

**Follow-ups**
- An email assistant summarizes inbox and can send mail. Design it so that a malicious email cannot exfiltrate other emails.
- How do you test injection resistance measurably?
- What is the confused-deputy problem in this setting?

---

## Q8.2 Jailbreaks and content-safety: how do you approach them in a product?

**Interviewer is testing:** Practical safety engineering without hand-waving.

**Expected answer / key points**
- Jailbreak classes: role-play/persona, obfuscation/encoding, multi-turn escalation, many-shot, adversarial suffixes, translation tricks, tool-mediated attacks.
- Layers: provider/model-level safety training, input and output classifiers, system-prompt policy, conversation-level monitoring (multi-turn), rate limiting and abuse detection, account-level enforcement, human escalation.
- Red teaming: internal and external; automated attack generation; track attack success rate over time.
- Response policy: refuse with helpful redirection; avoid over-refusal (measure false-refusal rate).
- Governance: policy definitions, incident response, disclosure, reporting channels.

**Senior/Lead considerations**
- Define product-specific harms (a children's app differs from an internal coding tool).
- Balance safety and usefulness; measure both attack success and over-refusal.
- Prepare an incident playbook (kill switch, patch prompts/classifiers, communicate).

**Common mistakes**
- Single keyword blocklist.
- No monitoring on multi-turn patterns.

**Follow-ups**
- Users report a new jailbreak on social media; what is your response timeline?
- How do you evaluate over-refusal?

---

## Q8.3 Data leakage and PII protection

**Interviewer is testing:** Privacy engineering across the AI data flow.

**Expected answer / key points**
- Leakage paths: prompts sent to third-party providers, logs/traces, caches, vector stores, fine-tuning data (memorization), model outputs revealing training or context data, system prompt extraction, cross-tenant contamination, embeddings inversion risks, analytics pipelines.
- Controls: **data classification** and policy on what may go to which model/provider; PII detection and redaction/pseudonymization before external calls (reversible tokenization where the answer needs it); zero-data-retention provider agreements; private endpoints; encryption; retention and deletion (including derived data); access control and audit on logs; differential handling of training/eval data (consent, minimization).
- Output-side DLP: scan responses for secrets/PII; canary strings to detect leakage.
- Regulatory framing: GDPR/CCPA (rights to access/delete), HIPAA, PCI, sector rules; DPIAs.

**Senior/Lead considerations**
- Deletion requests must reach vector stores, caches, backups, eval sets and fine-tuning datasets.
- Data minimization: send the least context necessary.
- Distinguish provider retention/training policies by product tier and contract; verify rather than assume.

**Common mistakes**
- Sending raw customer data to any API.
- Logging prompts with PII indefinitely.
- Training on production logs without a consent/governance path.

**Follow-ups**
- A customer asks you to delete all their data; enumerate every place it might live.
- How would you detect that a fine-tuned model memorized customer data?

---

## Q8.4 Tool abuse and excessive agent permissions

**Interviewer is testing:** Applying least privilege and authorization rigor to agents.

**Expected answer / key points**
- Risks: agents with broad credentials, tools that accept raw SQL/shell/URLs, chained tools enabling privilege escalation, confused deputy, abuse of paid or destructive tools, silent data exfiltration through allowed channels.
- Controls: **per-user delegated authorization** (act with the user's rights), per-task scoped short-lived tokens, allowlisted operations with typed parameters (no free-form shell/SQL in production), read-only defaults, approval for high-impact actions, sandboxing (containers, gVisor/Firecracker-style isolation, egress controls), rate/spend limits, policy engines (OPA/Cedar-style) evaluated outside the model, full audit logging.
- Design principle: **the model proposes; deterministic code disposes**. Authorization checks happen in the tool layer with the user's identity, not in the prompt.

**Senior/Lead considerations**
- Classify tools by risk tier and attach controls per tier.
- Review permission creep over time (periodic access reviews for agents).

**Common mistakes**
- Agent runs with a superuser token.
- Validation performed only by prompting ("don't drop tables").

**Follow-ups**
- Give a text-to-SQL agent access to analytics safely. (Read-only role, row-level security, allowlisted schemas, query cost/time limits, parser-level checks.)
- How would an attacker chain two individually safe tools into a harmful action?

---

## Q8.5 Tenant isolation in multi-tenant AI systems

**Interviewer is testing:** End-to-end isolation thinking.

**Expected answer / key points**
- Isolate at every layer: identity, API, retrieval indexes/namespaces, caches (prompt/semantic/KV prefix), memory stores, logs/traces, fine-tuned adapters/models, evaluation datasets, queues, and GPU processes if required.
- Strength levels: logical (tenant IDs and filters), namespace/partition, dedicated resources (per-tenant index, node pool, or cluster) for regulated or high-value tenants.
- Prevent cross-tenant side channels: shared prefix caches and timing, shared fine-tuned weights, shared embedding-based semantic caches.
- Testing: automated cross-tenant probes, canary records, periodic penetration tests, audit of query filters.
- Operational: per-tenant encryption keys (BYOK), per-tenant quotas, incident isolation.

**Senior/Lead considerations**
- Pick isolation tiers by sensitivity and cost; offer dedicated tiers for enterprise customers.
- Code paths that build filters are critical; enforce via a data-access layer, not scattered checks.

**Common mistakes**
- Filters applied in application code inconsistently.
- Shared cache keys lacking tenant scope.

**Follow-ups**
- How do you prove isolation to a customer's security team?
- How do you handle a shared base model with tenant-specific adapters safely?

---

## Q8.6 Supply-chain risks for AI systems

**Interviewer is testing:** Awareness of model, data and dependency provenance risks.

**Expected answer / key points**
- Risks: malicious or backdoored model weights, unsafe serialization (pickle) executing code on load, poisoned datasets or fine-tuning data, compromised packages (typosquatting, dependency confusion), malicious MCP servers/tools/plugins, tampered container images, poisoned embeddings/indexes, model-hub impersonation.
- Controls: use trusted registries; verify hashes/signatures and provenance; prefer safe formats (for example safetensors); scan artifacts; pin and lock dependencies; SBOMs; private mirrors; sandbox model loading; review third-party tools/prompts; staged evaluation of new models for safety/behavior regressions including backdoor probes; restrict egress from inference nodes.
- Governance: approved-model list, license review, model cards, risk assessments, change control.

**Senior/Lead considerations**
- Treat model weights as executable-adjacent artifacts; apply software supply-chain practices (SLSA-style provenance).
- Data poisoning defenses: source vetting, anomaly detection, data lineage.

**Common mistakes**
- Downloading community checkpoints straight into production.
- No provenance on training/fine-tuning data.

**Follow-ups**
- A popular open model on a hub is reported compromised; what is your response?
- How do you vet a third-party MCP server before enabling it?

---

## Q8.7 Secure MCP and tool execution (deep dive)

**Interviewer is testing:** Applied security for the tool ecosystem.

**Expected answer / key points**
- Threats: **tool poisoning** (hidden instructions in tool metadata), **rug pull/tool shadowing** (definition changes or a malicious tool overriding a trusted one), **confused deputy** and token passthrough, over-privileged scopes, SSRF via tool-fetched URLs, command injection in local servers, credential theft, data exfiltration through tool arguments, cross-server context leakage (one server reading data meant for another).
- Controls: pin and hash tool definitions and alert on change; human review of new tools; per-server isolation and least-privilege scopes; **OAuth with audience-restricted tokens** (no passthrough of tokens not issued for the server); consent screens per tool and per scope; sandbox local servers (containers, restricted filesystem/network); egress allowlists; input validation and output size limits; sanitize/label tool outputs as untrusted; separate servers by trust domain; audit logs with user, agent, tool, args, result.
- Operational: registry, vulnerability management, rapid revocation.

**Senior/Lead considerations**
- Multi-server setups create cross-tool attack paths; consider policy on which servers can co-exist in one context.
- Protocol and ecosystem practices evolve quickly; track the spec's security guidance.

**Common mistakes**
- Auto-approving all tool calls.
- Trusting descriptions of tools from third-party servers.

**Follow-ups**
- How do you defend against a malicious server that tries to make the agent call a sensitive tool on a different server?
- How do you revoke a compromised server across all agents within minutes?

---

## Q8.8 Access control for models and applications, and the OWASP LLM Top 10

**Interviewer is testing:** Governance frameworks and knowing the risk taxonomy.

**Expected answer / key points**
- **Access control layers:** who can call which model (RBAC/ABAC per team/project/data class), who can change prompts/tools/policies (change control, approvals), who can view traces/logs, who can deploy models, who can access fine-tuned artifacts, environment separation, key management, and service-to-service identity.
- **OWASP Top 10 for LLM applications** (2025 edition; verify current list): prompt injection; sensitive information disclosure; supply chain; data and model poisoning; improper output handling; excessive agency; system prompt leakage; vector and embedding weaknesses; misinformation; unbounded consumption.
- Map each risk to controls, owners, tests, and monitoring; maintain a risk register.
- Frameworks to reference: NIST AI RMF, ISO/IEC 42001, MITRE ATLAS, and applicable regulation (for example the EU AI Act risk tiers).

**Senior/Lead considerations**
- Improper output handling: treat model output as untrusted (XSS, SQL injection, command execution when passed downstream).
- Unbounded consumption: token limits, rate limits, budgets (denial-of-wallet).
- Responsible AI processes: model/system cards, impact assessments, human oversight, appeal paths, red-team results.

**Common mistakes**
- Treating model output as trusted input to other systems.
- Security review only at launch, not with each model/tool change.

**Follow-ups**
- Walk through "improper output handling" with a concrete exploit and its fix.
- What is your release checklist for a new agent feature from a security standpoint?

---

## Q8.9 Responsible AI: fairness, transparency, human oversight, and compliance in a product context

**Interviewer is testing:** Ability to operationalize responsible AI rather than recite principles.

**Expected answer / key points**
- Fairness: evaluate performance across user segments and languages; bias probes in evals; mitigation via data, prompts, thresholds; monitor for drift.
- Transparency: disclose AI use, show sources/uncertainty, explain limits, label generated content; provide user control and feedback/appeal.
- Human oversight: HITL for consequential decisions; auditability; clear accountability.
- Compliance: risk classification, documentation (model cards, data sheets), logging for audit, DPIAs, retention; regional obligations.
- Operationalization: policy-as-code, launch gates (safety review, red-team sign-off), incident response, periodic re-audits.

**Senior/Lead considerations**
- Integrate responsible-AI checks into CI/CD and eval suites rather than one-off reviews.
- Escalation path for ethical concerns; empower teams to block launches.

**Common mistakes**
- Principles without measurable gates.
- No plan for post-launch monitoring of harms.

**Follow-ups**
- How would you detect and mitigate performance gaps across languages?
- What would make you delay a launch?

# Appendix

## How does a large language model generates tokens?
1. Token IDs → Embeddings
Your prompt is split into tokens and each token is converted into a dense vector.

2. Transformer blocks
These embeddings pass through many transformer layers. Each layer:

Self-attention: creates Q/K/V and determines which previous tokens are relevant. RoPE provides positional information, while the causal mask prevents looking at future tokens.
FFN/SwiGLU: transforms the attended representation to capture higher-level patterns.
Residual connections + normalization: stabilize information flow and training.

3. Final prediction
After the last block, final normalization → LM head converts the representation into logits, one score per vocabulary token.

4. Sampling
Logits become probabilities using softmax, and a token is selected using the decoding strategy (greedy, temperature, top-p, etc.).
