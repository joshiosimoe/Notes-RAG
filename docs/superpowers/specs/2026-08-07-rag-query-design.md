# Vault Ingestion, Query Path, Demo Split, and Frontend — Design

Extends `2026-08-02-rag-design.md`. That document designed the whole system; this
one designs the half that was never built — build-order steps 6 through 8 — and
corrects four of its assumptions that turned out to be false against the live
account.

Decision numbering continues from that document, which ended at 18.

---

## 1. Goal

Make the system answerable. Today the indexer runs every five minutes and
publishes an index nobody can query. At the end of this work:

- Notes from the Obsidian vault are indexed alongside the video corpus.
- A question returns a grounded answer with citations that deep-link to the
  source — a timestamped video URL, or a note that opens in Obsidian.
- A signed-out visitor gets a working public demo over the video corpus.
- A signed-in owner gets the same interface over everything.

Two constraints carry forward unchanged: **≤ ~$5/month**, and the project is
**portfolio-facing**, so a live URL has real value.

---

## 2. Corrections to the 2026-08-02 design

Each of these was checked against the deployed system, not assumed.

**2.1 The notes are not in a GitHub repo.** Decision 3 chose a GitHub source on
the grounds that "notes already live in a private GitHub repo, so a private S3
bucket in the same account is not a larger trust step." They do not. Both vaults
live in OneDrive (`joshiosimoe`, 34 markdown files; `Agentic-OS`, 2) and neither
directory is a git repository. The rationale for the GitHub source evaporates
with its premise — see decision 19.

**2.2 `full.db` and `public.db` are currently identical.** Both hold 54 chunks,
all `corpus='video'`, `vault_id` NULL throughout. Step 6 was skipped, so no
markdown has ever been indexed. Until notes land, the Cognito-gated `/api` path
would protect nothing the public demo does not already serve, and the note half
of §4.2's citation scheme has nothing to cite. This is why the vault source is
sequenced before the query path rather than after it.

**2.3 §4.2's Haiku request shape is wrong for Haiku 4.5.** It specifies
`thinking={"type": "disabled"}` and no `temperature` / `top_p` / `top_k`,
citing §9's warning that those return 400. That warning describes Opus 4.7+ and
Sonnet 5. On Haiku 4.5 the thinking parameter takes
`{"type": "enabled", "budget_tokens": N}` and is simply omitted for no thinking;
sampling parameters are accepted; and `output_config.effort` errors. See
decision 27.

**2.4 The Bedrock client surface has moved.** `AnthropicBedrockMantle` is now the
preferred client (Messages-API endpoint, model IDs prefixed `anthropic.`);
`AnthropicBedrock` is the legacy `bedrock-runtime` `InvokeModel` path, which is
where the inference-profile ID `us.anthropic.claude-haiku-4-5-20251001-v1:0`
lives. Which of the two this account can actually reach in `us-east-2` is a
verification item (§12), not an assumption.

**2.5 There is no concurrency ceiling to lean on.** `infra/indexer.tf` records
that this account's Lambda concurrency quota is 10, and AWS refuses any
reservation that would drop unreserved capacity below 10 — so
`reserved_concurrent_executions` is impossible at any value. A public
unauthenticated endpoint that calls a paid model therefore has no infrastructure
backstop, which makes the spend guard in decision 23 load-bearing rather than
defensive.

---

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| 19 | **Notes reach the indexer through an S3 prefix, synced from the vault — not the GitHub contents API** | Decision 3's premise is false (§2.1). The S3 lister, ETag manifest diff, markdown chunker, and wikilink parsing are already built, tested, and deployed; the GitHub path is a plan's worth of new code (compare API, per-repo commit watermarks, a token in Secrets Manager, a second read path for files over the 1MB contents-API cap) for a 34-file vault that is not in git. Cost: renames arrive as delete+add rather than as renames — which the indexer already handles correctly. |
| 20 | **The indexer takes a JSON source list, replacing `SOURCE_BUCKET` + `SOURCE_PREFIXES`** | Decision 3 always called for a source list; the single-bucket pair was a shortcut taken when only one source existed. Adding a second ad-hoc pair recreates precisely the drift between the IAM grant and the env var that `07b819a` had to fix. One list, one Terraform variable, one IAM derivation. |
| 21 | **The vault source is built before the query path** | It is the only change that makes the rest meaningful: it gives the full/public split something to separate, gives note citations something to cite, and delivers the second half of the stated goal. Building the query path first would ship note-citation code that has never run against real data. |
| 22 | **CloudFront → API Gateway HTTP API → two Lambdas** | One domain, so no CORS. The HTTP API supplies two things that would otherwise be application code: a native JWT authorizer against the Cognito pool — keeping JWKS fetching, caching, and signature verification, the actual security boundary, out of code we maintain — and per-route throttling, which is the burst half of the spend guard. ~$1 per million requests. |
| 23 | **Demo spend is capped by conditional DynamoDB counters: monthly and daily** | No reservation is possible (§2.5), so the cap has to live in the request path. A monthly cap is the one that bounds the bill; a daily sub-cap stops one bad day consuming the month. Both are Terraform variables. |
| 24 | **Citations are assembled server-side from context indices; the model never emits a URL or timestamp** | A fabricated deep link is worse than no link: it looks authoritative and fails silently. The prompt numbers the context blocks, the model cites indices, and the handler maps indices to citations built from stored chunk metadata. Every emitted link is therefore real by construction. |
| 25 | **Below a retrieval distance threshold, refuse without calling Bedrock** | An out-of-corpus question has no grounded answer available, so generating one invites a hallucination and pays for the privilege. Refusing early is both the cheapest and the most accurate response. |
| 26 | **Artifact freshness is checked per invocation with a HEAD on the ETag** | The indexer republishes every five minutes; a warm container that downloaded once serves an arbitrarily stale index for its entire life, with no symptom. A ~20ms HEAD per invocation buys correctness on a request path already spending seconds in Bedrock. |
| 27 | **Haiku 4.5 request shape: omit `thinking`, omit `effort`, omit sampling parameters** | Omitting `thinking` is how thinking is disabled on this model family; `effort` errors on Haiku 4.5; sampling parameters are legal here but add a knob with no evaluation behind it. Corrects §2.3. |
| 28 | **Single-turn, non-streaming, k=6** | Restates decision 18 in query terms. k=6 is what the eval baseline (recall@6 1.000, MRR 0.967) was measured at, so changing it silently invalidates the only retrieval number the project has. |
| 29 | **`AnthropicBedrock` — the legacy `bedrock-runtime` client — over an inference profile, never a bare model id** | Settled by real calls, not docs (§12.1). Haiku 4.5 has **no in-region availability in `us-east-2`**, so the bare `anthropic.claude-haiku-4-5-20251001-v1:0` is rejected outright (`on-demand throughput isn't supported`) — a profile is mandatory, not a preference. `AnthropicBedrockMantle` is the forward-looking client and its endpoint does resolve in `us-east-2`, but every call 403s on `bedrock-mantle:CreateInference`, which this account lacks; the legacy path needs no new IAM action beyond the `bedrock:InvokeModel` the indexer already uses. `global.` works too and is 10% cheaper, but `us.` is chosen for US-only routing and a four-ARN rather than ~thirty-ARN policy, at ~$0.30/month (§10). |

---

## 4. Plan decomposition

Four plans in order. Each ends deployed and verified against the live account
rather than merely merged.

| Plan | Delivers | Done when |
|---|---|---|
| 3 | Vault ingestion via S3 | Notes present in `full.db`, absent from `public.db` |
| 4 | Retrieval, generation, citations | `aws lambda invoke` returns a grounded, cited answer |
| 5 | CloudFront, API Gateway, Cognito, spend guard | The public demo URL answers; `/api` rejects unauthenticated requests |
| 6 | Vite/React frontend | Signed-out demo and signed-in search both work in a browser |

---

## 5. Plan 3 — vault ingestion

### 5.1 Storage

A new bucket, `notes-rag-source-<account-id>`: versioned, SSE, public access
blocked. Not the index bucket — the indexer holds write access there, and a
source the consumer can overwrite is not a source.

Layout:

```
s3://notes-rag-source-<account-id>/notes/<vault_id>/<path within vault>.md
```

The `<vault_id>` path segment is load-bearing. Every chunk's `vault_id` is
currently NULL, and `obsidian://open?vault={vault_id}&file={path}` cannot be
constructed without it.

### 5.2 Sync

`scripts/sync_vault.sh` wraps:

```bash
aws s3 sync "$VAULT_DIR" "s3://$BUCKET/notes/$VAULT_ID/" --delete \
  --exclude '.obsidian/*' --exclude '.trash/*'
```

`--delete` is not optional: the manifest diff turns a removed object into a chunk
deletion, so without it a note deleted in Obsidian stays answerable forever.

### 5.3 Source list

`SOURCE_BUCKET` and `SOURCE_PREFIXES` are replaced by one JSON-encoded list:

```json
[
  {"bucket": "videovaultstack-...", "prefixes": ["summaries/", "transcripts/"],
   "corpus": "video"},
  {"bucket": "notes-rag-source-...", "prefixes": ["notes/joshiosimoe/"],
   "corpus": "note", "vault_id": "joshiosimoe"}
]
```

`vault_id` is required on `corpus: "note"` entries and absent on every other
corpus — it exists to build `obsidian://` citations, and a video chunk has no
vault to open.

Terraform derives both the env var and the IAM resource ARNs from the same
variable, preserving the property `07b819a` established: the grant and the code
cannot drift apart.

### 5.4 What this changes downstream — nothing

Markdown chunks carry `corpus="note"`, so `public.db`'s existing `corpus='video'`
filter excludes them with no new code, and `copy_filtered`'s backlink stripping
(added in `07b819a`) stops note names leaking through the graph. **The
full/public split stops being decorative at this point** and starts separating
real content.

### 5.5 One thing that is free now and expensive later

`vault_id` is embedded in every markdown chunk's context and therefore in its
`content_hash`. Changing it later is a 100% cache miss and a full re-embed of the
note corpus. Since zero markdown is indexed today, choosing the final value now
costs nothing.

---

## 6. Plan 4 — query path

### 6.1 Layout

```
src/notes_rag/query/retrieve.py     embed question, search, threshold
src/notes_rag/query/generate.py     prompt assembly, Bedrock call, answer parse
src/notes_rag/query/handler.py      Lambda entrypoint over full.db
src/notes_rag/demo/handler.py       same path over public.db
```

The demo handler differs only in which artifact key it reads. Isolation is
enforced by its IAM role (decision 5), not by this code — a bug here cannot leak
`full.db` because the role cannot read it.

### 6.2 Request path

1. HEAD the artifact; re-download to `/tmp` only if the ETag changed
   (decision 26).
2. Embed the question with Titan v2 — the existing `BedrockEmbedder`.
3. `store.search(vector, k=6)`.
4. If the best hit is worse than the distance threshold, return a refusal with
   zero citations and stop (decision 25). The threshold is a config value, and
   its initial setting is chosen by measuring the distance distribution of known
   hits in the golden set against deliberately out-of-corpus questions — not
   guessed, since a threshold set too tight refuses answerable questions and one
   set too loose defeats the decision.
5. Assemble a prompt: system instructions on grounding and citation, user turn
   carrying the question plus numbered context blocks.
6. Call Haiku 4.5 (decision 27).
7. Map cited indices to citation objects (decision 24).

### 6.3 Response contract

```json
{
  "answer": "...",
  "citations": [
    {"kind": "video", "title": "...", "url": "https://...&t=1120",
     "start_seconds": 1120, "source_path": "summaries/...", "chunk_id": "..."},
    {"kind": "note", "title": "...", "url": "obsidian://open?vault=...&file=...",
     "source_path": "notes/joshiosimoe/...", "chunk_id": "..."}
  ],
  "corpus_scope": "public" | "full",
  "model": "...",
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

`corpus_scope` is echoed so the frontend can say plainly which index answered,
rather than leaving a demo user to wonder why their question about notes found
nothing.

### 6.4 Failure modes

| Condition | Response |
|---|---|
| Bedrock throttled | 503 with `Retry-After` |
| Embedding call fails | 502 — the question was never searchable |
| Artifact missing from S3 | 503; the indexer restores it within one tick |
| No hit clears the threshold | 200 with a refusal and zero citations |
| Model cites an index not in context | Drop that citation, keep the answer, log a warning |

The last row matters: a citation index the model invented is the one failure
decision 24 cannot prevent, only contain.

---

## 7. Plan 5 — HTTP surface, auth, spend guard

### 7.1 Routing

CloudFront default behaviour serves the static site from S3 via OAC. `/api/*`
and `/demo/*` route to an HTTP API origin. One domain, so the frontend needs no
CORS handling and the demo is a single clean link.

HTTP API routes:

| Route | Auth | Throttle |
|---|---|---|
| `POST /api/query` | Cognito JWT authorizer | default |
| `POST /demo/query` | none | burst and rate limited |

### 7.2 Auth

A Cognito user pool with the hosted UI, Authorization Code + PKCE
(`oidc-client-ts` on the client — implicit flow is deprecated and hand-rolling
PKCE is the error-prone path). Single user.

### 7.3 Spend guard

DynamoDB table `notes-rag-demo-budget`, partition key a period string:

```
month#2026-08     count=N
day#2026-08-07    count=N   (TTL: 48h)
```

Before each Bedrock call the demo handler issues two conditional updates —
`ADD count 1` with `ConditionExpression` on the cap. A `ConditionalCheckFailed`
returns 429 with a plain "demo quota reached" message. Both caps are Terraform
variables.

**Check-then-call is deliberate.** A generation that fails after the increment
still consumes a unit. The alternative — reserve, then refund on failure — adds
a distributed-transaction shape to recover fractions of a cent, and its failure
mode (a lost refund) is a cap that drifts *looser* over time. A cap that drifts
slightly tighter is the safer direction to be wrong in.

### 7.4 IAM

The demo Lambda's role gets `s3:GetObject` on `index/public.db` and nothing else
in that bucket — decision 5's boundary, unchanged. Both roles get
`bedrock:InvokeModel` scoped to the embedding and generation model ARNs.

The generation grant is **not** the single-ARN shape `infra/iam.tf` uses for
Titan. Invoking through an inference profile (decision 29) authorizes against two
resource types at once, and a policy naming only one of them fails closed:

```
arn:aws:bedrock:us-east-2:<acct>:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0
arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0
arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0
arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0
```

The three foundation-model ARNs are the `us.` profile's destination regions as
published on the model card; note they carry no account id. Choosing `global.`
instead replaces these with ~30 regional ARNs. This is the part of the profile
choice that costs maintenance rather than money, and it is why the Terraform
should derive the list from a variable rather than hardcode it inline.

---

## 8. Plan 6 — frontend

Vite + React + TypeScript, four components: query box, answer pane, citation
card, auth button. Signed out posts to `/demo/query`; signed in posts to
`/api/query` with a bearer token.

Non-streaming (decision 18 — Python Lambda cannot use response streaming), so the
UI shows staged `retrieving → generating` progress text over the ~3–5s rather
than an undifferentiated spinner.

Deploy is `npm run build`, `aws s3 sync dist/`, CloudFront invalidation — in a
script, because a deploy that skips the invalidation serves the previous bundle
with no error and no diff.

---

## 9. Testing and evaluation

A fake Bedrock client alongside the existing `FakeEmbedder`, so generation tests
are deterministic and cost nothing.

Unit coverage concentrated where a bug is silent rather than loud:

- Index-to-citation mapping, including the invented-index case.
- The refusal threshold, on both sides of the boundary.
- The budget counter's conditional logic against a stubbed DynamoDB.
- That the demo handler resolves its artifact key to `index/public.db` and never
  to `index/full.db`, asserted on the key the handler asks for. IAM is the real
  enforcement and is not exercisable in a unit test, so this covers the layer
  that is.
- ETag staleness: a changed ETag re-downloads, an unchanged one does not.

Integration tests stay opt-in (`pytest -m integration`) and invoke the deployed
demo Lambda.

The retrieval eval remains the CI gate, unchanged and deterministic.
`eval/judge.py` (decision 16) lands in Plan 4 — Haiku scoring groundedness and
citation accuracy behind `--judge` — because generation is otherwise the first
component in the system with no measurement at all.

---

## 10. Cost

Per demo answer, estimated: ~5.5K input tokens (six chunks plus prompt and
question) and ~400 output. Query embedding via Titan v2 is negligible.

Bedrock partner pricing is now verified (§12.2) from the AWS Price List bulk API
— `AmazonBedrockFoundationModels`, `us-east-2`, effective 2026-07-01. Anthropic
models are billed through Marketplace, which is why they are absent from the
`AmazonBedrock` price list and from the rendered pricing page. Rates depend on
which inference profile is used:

| Per MTok | `global.` profile | `us.` geo profile |
|---|---|---|
| Input | $1.00 | $1.10 |
| Output | $5.00 | $5.50 |
| Cache read | $0.10 | $0.11 |
| Cache write (5m) | $1.25 | $1.375 |
| **Per answer** | **$0.0075** | **$0.00825** |

`global.` matches first-party Haiku 4.5 exactly, so the 08-02 estimate was right
for that profile and 10% low for `us.`.

| Line | Monthly (`global.`) | Monthly (`us.`) |
|---|---|---|
| Existing indexer, S3, Lambda | ~$0.20 | ~$0.20 |
| Generation at a 400-answer cap | ~$3.00 | ~$3.30 |
| DynamoDB (2 writes per answer, on demand) | ~$0.00 | ~$0.00 |
| API Gateway, CloudFront, Cognito | ~$0.00 — inside free tiers | ~$0.00 |
| **Total at the cap** | **~$3.20** | **~$3.50** |

Both clear the ~$5 ceiling, so the profile choice was never a budget question —
the cap variables remain the control. It was a residency and IAM question, and
**`us.` is chosen** (decision 29): it routes only to `us-east-1`/`us-east-2`/
`us-west-2`, keeping vault notes in US regions, and needs four IAM ARNs against
`global.`'s ~30 — or a wildcard, which would give up the scoping decision 5 and
§7.4 rely on. The premium is ~$0.30/month at the cap. Both profiles were
confirmed working from `us-east-2` with a real call.

**Budget at the cap is therefore ~$3.50/month**, and §7.3's cap variables should
be read against that figure rather than the 08-02 spec's ~$3.20.

---

## 11. Open items deferred by choice

- **GitHub vault source.** Decision 19 replaces it for now. If the vault later
  moves into git, the source list from decision 20 accommodates a GitHub entry
  with no query-side rework.
- **The `Agentic-OS` vault** (2 files) is not indexed initially. Adding it is one
  more source-list entry.
- **Multi-turn conversation.** Single-turn in v1; adding history changes the API
  contract and the frontend's state model, not the retrieval or generation code.
- **Retrieval mix** between summary and transcript chunks — still a config value,
  still waiting on a golden set large enough to measure it.
- **Wikilink expansion** — data model built, behaviour still deferred until the
  eval can detect whether it helps.
- **Class-material parsers** (docx, pptx, xlsx, pdf) — build-order step 9,
  untouched here.

---

## 12. Verification required before building

1. ~~**Which Bedrock client this account can reach in `us-east-2`**~~ —
   **settled 2026-08-07, decision 29.** Six combinations were called for real
   against `us-east-2`; exactly two returned an answer:

   | Client | Model id | Result |
   |---|---|---|
   | `AnthropicBedrock` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | **200** |
   | `AnthropicBedrock` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | **200** |
   | `AnthropicBedrock` | `anthropic.claude-haiku-4-5-20251001-v1:0` | 400 — on-demand throughput unsupported |
   | `AnthropicBedrock` | `anthropic.claude-haiku-4-5` | 400 — invalid model identifier |
   | `AnthropicBedrockMantle` | all three id forms | 403 — `bedrock-mantle:CreateInference` denied |

   `ListFoundationModels` corroborates: every Anthropic model in `us-east-2`
   reports `INFERENCE_PROFILE` as its only supported inference type, and the AWS
   model card marks `us-east-2` in-region as unsupported for Haiku 4.5.
   `generate.py` therefore imports `AnthropicBedrock`. The Mantle 403 is an IAM
   gap rather than an availability one, so this is revisitable — but not for
   free, and there is no reason to pay for it yet.
2. ~~**Bedrock partner pricing for Haiku 4.5**~~ — **settled 2026-08-07, §10.**
   `global.` is $1.00/$5.00 per MTok, `us.` is $1.10/$5.50; the cap arithmetic
   holds on either. Josh chose `us.` for US-only routing and the smaller IAM
   surface, at ~$0.30/month more. Budget at the cap moves from ~$3.20 to ~$3.50.
3. **Cognito free-tier limits** for a single-user app.
4. **Deployer IAM.** New permissions are needed for DynamoDB, Cognito,
   CloudFront, and API Gateway. The `terraform-deployers` group is at AWS's cap
   of 10 managed policies, so these extend the existing inline
   `NotesRagSchedulerDeploy` policy — the same workaround the scheduler deploy
   needed.

Not designing around prompt caching: Haiku 4.5's minimum cacheable prefix is 4096
tokens and the system prompt will not approach it, so a `cache_control` marker
would silently cache nothing.
