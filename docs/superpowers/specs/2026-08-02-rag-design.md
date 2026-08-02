# RAG over Video Vault + Obsidian Vaults + Class Materials — Design

**Status:** approved, not started.
**Date:** 2026-08-02.
**Input:** `docs/rag/BRIEF.md`. Sections marked DECIDED there are constraints and are
not revisited here. This document closes every item in the brief's §10 OPEN list,
plus IaC, vault layout, Bedrock region, embedding cache, and the frontend stack.

---

## 1. Goal

Ask questions across a semester of study material and get answers with citations
back to the source: for videos, a deep link to the exact timestamp; for vault
notes, a link that opens the note.

Three corpora:

1. **Video summaries** — machine-written JSON in S3, written by Video Vault. Immutable.
2. **Obsidian notes** — hand-authored markdown in one or more private GitHub repos. Mutable.
3. **Class materials** — docx, pptx, xlsx, pdf. Uploaded to S3.

Two hard constraints carried from the brief: **≤ ~$5/month**, and the project is
**portfolio-facing** — a demoable URL has real value.

---

## 2. Decisions

Every decision below was made in the design conversation. Where a decision
contradicts or sharpens the brief, that is called out explicitly.

| # | Decision | Rationale |
|---|---|---|
| 1 | **All corpora indexed in AWS; auth on the real endpoint; public demo scoped to videos** | Notes already live in a private GitHub repo, so a private S3 bucket in the same account is not a larger trust step. Resolves the §7 goal conflict without splitting into two code paths. |
| 2 | **`sqlite-vec` file in S3, behind a `VectorStore` interface** | ~20k chunks × 1024 dims ≈ 82MB vectors + ~30MB text. At that size ANN is unnecessary — brute-force is milliseconds. SQL gives metadata filtering for free. Interface keeps a pgvector benchmark available later. |
| 3 | **Ingester takes a config list of `{vault_id, repo, prefix}` sources** | The user said "vault(s)". Repos and Obsidian vaults are independent concepts; a source list works for one repo, two repos, or subfolders. Layout can be decided later with no rework. |
| 4 | **EventBridge Scheduler every 5 min + on-demand invoke; `reserved_concurrency = 1`** | The bottleneck is git push cadence, not AWS. A 5-min cron plus a manual trigger beats a webhook here because it needs no public endpoint, no HMAC, no SQS, and cannot race on the index artifact. |
| 5 | **Two artifacts: `full.db` and `public.db`** | The demo Lambda's IAM role can read only `public.db`. Demo isolation is enforced at the IAM boundary rather than by a query predicate, so a filter bug cannot leak personal notes. |
| 6 | **CloudFront + S3 static site; Cognito JWT; signed out = demo mode** | One frontend, two backends. Signed out hits `/demo/*` over `public.db`; signed in hits `/api/*` over `full.db`. |
| 7 | **Terraform + zip build script** | Portfolio covers CDK (Video Vault) and Terraform, each with a defensible reason. The §8 bundling objection dissolves because `sqlite-vec` ships prebuilt wheels — `pip install --platform manylinux_2_17_x86_64 --only-binary=:all: -t build/` needs no Docker. |
| 8 | **Embed both summaries and transcripts; `chunk_type` metadata; retrieval mix is config** | Storage is ~80MB and pennies. Transcripts are the only source of detail the summarizer drops. Making the mix a config value turns it into an eval experiment rather than a guess. |
| 9 | **Haiku 4.5, model ID as config** | Grounded answering over retrieved chunks is the easiest place in the system to downgrade (brief §10.12 agrees). ~$0.55/mo at 100 queries vs ~$1.70 for Sonnet 4.6. One env var to upgrade if the eval says otherwise. |
| 10 | **Submit the Bedrock Anthropic use-case form for `us-east-2`** | One-time form that permanently deletes the cross-region inference profile and its three-region `InvokeModel` grant. §9 warns one call can slip through before the gate applies — verify with several calls, not one. |
| 11 | **`amazon.titan-embed-text-v2:0`, 1024 dims** | All of §2's sizing assumes it. Not an Anthropic model, so the use-case form does not apply. **Must make one real `InvokeModel` call before building on it** — §9: listing is not entitlement. |
| 12 | **Per-format chunkers behind one interface + shared normalizer** | Five source shapes with genuinely different natural boundaries, but one set of size rules. |
| 13 | **Class materials: S3 prefix + EventBridge rule, cron as backstop** | Binary files do not belong in the git vault — repo bloat, and GitHub's contents API caps at 1MB, which would force a second read path. The event is a best-effort accelerator; `reserved_concurrency = 1` drops burst overflow and the cron sweeps it up. |
| 14 | **xlsx indexed as per-sheet markdown tables, with a size ceiling** | Covers syllabus grids and reference tables. Oversized sheets are skipped rather than embedded as noise. |
| 15 | **Wikilinks parsed to metadata in v1; gated expansion after the eval exists** | The graph is captured from day one, so expansion is later a query-side change with no re-index. Shipping expansion before the golden set exists means shipping an unmeasurable precision risk. |
| 16 | **Golden-question eval in CI (retrieval-only) + LLM judge behind a flag** | Deterministic, free, hard-gating metrics on every commit; expensive groundedness/citation scoring on demand. ~$3/mo if the judge ran per-commit — 60% of the remaining budget, which decides cadence. |
| 17 | **Incremental embedding, full artifact rebuild; previous `full.db` is the cache** | See §3 below — this **corrects the brief**. |
| 18 | **Vite + React + TypeScript; non-streaming answers in v1** | A bundler is required for the OIDC library regardless, so React costs roughly a `package.json` over vanilla. Python Lambda cannot use response streaming. |

---

## 3. Correction to the brief: incremental embedding is mandatory

Brief §4 states a full re-embed costs ~11 cents and concludes: *"argue this on
latency and complexity, not cost."* That holds for a **one-time** rebuild. It does
not hold once the trigger fires every 5 minutes.

| Scenario | Rebuilds/month | Cost if each re-embeds everything |
|---|---|---|
| One-time re-index | 1 | $0.12 |
| ~20 note edits/day | ~600 | **~$72/month** |

Cost therefore returns as the deciding factor. Two separate concerns, only one of
which is expensive:

- **Embedding — must be incremental.** Re-embed only chunks whose `content_hash`
  changed. A note edit is ~5 chunks ≈ 2,500 tokens ≈ $0.00005.
- **Artifact assembly — rebuild fully every run.** Local file writes plus an S3
  upload, zero API calls. No reason to complicate it.

**The no-op path is load-bearing.** 8,639 of ~8,640 monthly runs find nothing
changed and must exit after the GitHub `compare` calls and S3 `ETag` diffs,
before downloading `full.db`.

**The cache is the previous `full.db`.** No separate store, no divergence.

---

## 4. Architecture

```
                    ┌─ EventBridge Scheduler (5 min) ──┐
GitHub vault repos ─┤                                   ├─► indexer Lambda
S3 class-materials ─┤─ EventBridge (S3 ObjectCreated) ──┤   reserved_concurrency=1
Video Vault bucket ─┘  aws lambda invoke (on demand) ───┘        │
                                                                 ▼
                                              ┌──────────────────────────────┐
                                              │ s3://rag/index/full.db       │  everything
                                              │ s3://rag/index/public.db     │  corpus=video only
                                              └──────────────────────────────┘
                                                     ▲                ▲
CloudFront ──► S3 static site                        │                │
           ├─► /api/*  ► query Lambda (Cognito JWT) ─┘                │
           └─► /demo/* ► demo  Lambda (no auth) ──────────────────────┘
                          IAM: s3:GetObject on public.db ONLY
```

### 4.1 Indexer

1. Read per-source watermarks (last-seen commit SHA per repo; ETag manifest per S3 prefix).
2. For each vault source: `GET /repos/{owner}/{repo}/compare/{base}...{head}` →
   `added` / `modified` / `removed` / `renamed` paths.
3. List `s3://rag/class-materials/`; diff ETags against the manifest.
4. List Video Vault's `summaries/` and `transcripts/`; diff against the manifest.
5. **If nothing changed anywhere, exit.** (~200ms, the overwhelmingly common case.)
6. Download previous `full.db`.
7. Chunk changed sources (per-format chunker → shared normalizer).
8. Per chunk: reuse the cached vector when `content_hash` matches, else call Titan v2.
9. Delete rows for removed paths and for the source side of renames.
10. Write `full.db`; write `public.db` as the `corpus = 'video'` subset.
11. Upload both; advance watermarks.

The indexer tolerates bulk updates: a `scripts/resummarize.py` pass rewrites the
whole `summaries/` prefix at once, which appears as a large ETag diff and is handled
by the same path. It never has to handle deletes in corpus 1.

### 4.2 Query

Cold start downloads the `.db` to `/tmp` (~1.5s for 130MB in-region). Warm
invocations reuse it.

Embed query → `sqlite-vec` top-k with metadata filters → assemble context →
Haiku 4.5 via classic `AnthropicBedrock` with `thinking={"type": "disabled"}`
and **no** `temperature` / `top_p` / `top_k` (§9 — these return 400).

Citations:
- Video: `{url}&t={start_seconds}`
- Note: both `obsidian://open?vault={vault_id}&file={path}` and a `github.com` blob URL

The demo Lambda runs the same code path against `public.db` and skips JWT verification.

### 4.3 Chunk schema

```sql
chunks(
  id, corpus, vault_id, source_path, chunk_type,   -- filters + demo scoping
  title, heading, text, content_hash,              -- content + cache key
  video_id, start_seconds, url,                    -- video citation
  links_to, backlinks,                             -- JSON arrays of paths; stored, unused in v1
  embedding                                        -- vec0, 1024 dims
)
```

`corpus` is what partitions `public.db` out of `full.db`. `content_hash` is the
embedding cache key. `links_to` is parsed from `[[wikilinks]]`; `backlinks` is that
relation inverted at build time.

### 4.4 Chunking

Two quality levers apply to every format:

**Contextual prefixing.** A video section summary averages ~440 chars (~110 tokens)
and embeds poorly alone. Every chunk is prefixed before embedding:
`{title} — {channel} — {section title}` for videos, `{vault_id} / {path} / {heading}`
for notes.

**Transcript chunks align to section boundaries.** `summary.sections[].start_seconds`
already partitions the video. Splitting the transcript on those boundaries means every
transcript chunk inherits a real timestamp — no separate windowing logic, and
citations stay deep-linkable.

Normalizer rules, applied to every chunker's output:
- Merge chunks under ~150 tokens into the adjacent chunk.
- Split chunks over ~800 tokens on paragraph boundaries.
- Apply the contextual prefix.
- Compute `content_hash` over the chunk text.

| Source | Boundary |
|---|---|
| Video summary | one chunk per `sections[]` entry, plus one for `verdict` + `tldr` + `takeaways` |
| Video transcript | caption segments grouped on `sections[].start_seconds` |
| Markdown note | headings |
| pptx | slide |
| docx | headings |
| pdf | headings, page fallback |
| xlsx | sheet, rendered as a markdown table; skip above a size ceiling |

**Dedupe.** Videos are ingested from S3 only. The `Video Vault/` prefix is excluded
from the vault ingester — the rendered markdown is strictly lossier than the JSON.
This is a path-prefix exclusion per source in the config list.

### 4.5 Frontend

Vite + React + TypeScript, built to `dist/`, `aws s3 sync`'d, CloudFront invalidated on
deploy. `oidc-client-ts` for Cognito Authorization Code + PKCE (implicit flow is
deprecated; hand-rolling PKCE is the error-prone path).

Four components: query box, answer pane, citation card, auth button.

**Non-streaming in v1.** Lambda response streaming is supported on Node.js managed
runtimes and custom runtimes, not the Python managed runtime. Everything else here is
Python. The UI shows a spinner with `retrieving → generating` progress text for the
~3–5s. Revisit only if it feels slow in real use.

---

## 5. Repo layout

```
src/chunkers/    video_summary, video_transcript, markdown, docx, pptx, xlsx, pdf, normalizer
src/store/       base.py (VectorStore: search / upsert / delete_by_path), sqlite_vec.py
src/sources/     github.py, s3.py
src/indexer/     handler.py, manifest.py
src/query/       handler.py, retrieve.py, generate.py
src/demo/        handler.py
eval/            questions.yaml, run.py (recall@k, MRR), judge.py (--judge)
infra/           *.tf
web/             Vite + React + TS
```

---

## 6. Evaluation

Golden questions anchored on **source span, not chunk ID** — chunk IDs are a function
of chunking config, so anchoring on them would invalidate the very test meant to
measure chunking changes. A retrieved chunk counts as a hit if it overlaps the
expected span.

```yaml
- id: q001
  question: "What did the Kubernetes video say about custom schedulers?"
  expects:
    - corpus: video
      video_id: dQw4w9WgXcQ
      start_seconds: [1120, 1400]
```

- **`eval/run.py`** — recall@k and MRR on retrieval alone. Deterministic, ~$0, runs in
  CI as a hard gate on every commit.
- **`eval/judge.py --judge`** — Haiku scores generated answers for groundedness and
  citation accuracy. Run before releases or while tuning the generation prompt.
  ~$0.15/run.

**Known limitation:** ~15 questions over 2 videos moves recall@k in 6.7% increments.
This detects "something broke," not "this is 3% better." Treat it as a regression
tripwire until the corpus grows, and state that rather than over-claiming the numbers.

---

## 7. Cost

| Line | Monthly |
|---|---|
| S3 storage (~350MB) + 8,640 LIST | ~$0.05 |
| Lambda (indexer + query + demo) | $0 — ~1,800 GB-s against 400,000 free |
| CloudFront + Cognito | $0 — inside free tiers |
| Bedrock embeddings (incremental) | ~$0.10 |
| Bedrock Haiku 4.5 @ 100 queries | ~$0.55 |
| **Total** | **~$0.70** |

Against a ~$5 ceiling. **Bedrock is partner-priced** — these figures are derived from
first-party rates and must be verified against the Bedrock pricing page. Headroom is
large enough that Sonnet 4.6 is a one-env-var upgrade if the eval demands it.

Explicitly rejected on cost (brief §6): OpenSearch Serverless (hundreds/month idle),
Aurora Serverless v2 pgvector at a 0.5 ACU floor (~$43/month).

---

## 8. Build order

1. `VectorStore` interface + `sqlite_vec` implementation. TDD, no AWS.
2. Chunkers + normalizer, against the 2 real videos already in the bucket.
3. Indexer: S3 video sources only, local `.db` output.
4. `eval/` harness + first ~15 golden questions. **Before any tuning.**
5. Terraform: bucket, indexer Lambda, scheduler, IAM.
6. GitHub vault source + watermarks.
7. Query Lambda: retrieval, then generation.
8. Cognito, CloudFront, static site, demo split.
9. Class-material parsers + EventBridge rule.
10. Gated wikilink expansion, measured on vs off.

Steps 1–4 need no AWS and produce the measurement harness before anything is tunable.

**Do first, both have lead time:**
- Submit the Bedrock Anthropic use-case form for `us-east-2`.
- Make one real `amazon.titan-embed-text-v2:0` `InvokeModel` call in `us-east-2`.

---

## 9. Open items deferred by choice

- **Vault repo layout** — one repo or two, vault-as-subfolder or vault-as-repo. The
  source-list config works for all of them; re-indexing costs ~11 cents if it changes.
- **Retrieval mix** between summary and transcript chunks — a config value, tuned once
  the eval set exists.
- **Wikilink expansion** — data model built in v1, behavior deferred until measurable.
- **pgvector benchmark** — the `VectorStore` interface leaves it available; not v1 scope.

---

## 10. Verification required before building

Each of these is an assumption this design rests on that has not been confirmed
against the live account:

1. `amazon.titan-embed-text-v2:0` returns a real embedding in `us-east-2`
   (§9: listing is not entitlement).
2. Bedrock's Anthropic use-case form is approved for `us-east-2`, verified with
   **several** calls — §9 warns one can slip through before the gate applies.
3. Bedrock partner pricing for Titan v2 and Haiku 4.5 matches the §7 estimates.
4. Cognito's current free-tier limits cover a single-user app.
5. Lambda response streaming is still unavailable on the Python managed runtime.
6. `sqlite-vec` manylinux wheels load correctly inside the Lambda runtime.
7. Re-measure §2's corpus sizing against the bucket once it holds more than 2 videos —
   the video mix may skew longer than the sampled 18.9-minute artifact.
