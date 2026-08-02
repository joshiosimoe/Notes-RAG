# RAG Project — Handoff Brief

**Status:** not started. Video Vault is deployed and producing the corpus.
**Written:** 2026-07-23, at the end of the Video Vault design session.
**Revised:** 2026-08-02 — Video Vault shipped, so the estimates in here were
replaced with measurements; scope expanded to hand-written vault notes.

> **To the agent picking this up:** this brief is an *input to a design
> conversation*, not a spec. Sections marked **DECIDED** are settled — treat
> them as constraints and don't relitigate them without a concrete reason.
> Sections marked **OPEN** are deliberately unresolved: interview the user about
> them before writing any plan. Do not fill them in yourself. The user wants to
> be consulted on design, not handed a finished blueprint.
>
> Start with the `superpowers:brainstorming` skill. This brief is background
> reading, not a substitute for the interview.

---

## 1. What this project is — SCOPE CHANGED 2026-08-02

A retrieval system over **three** corpora:

1. **YouTube video summaries** produced by Video Vault. Machine-written,
   structured JSON in S3, write-once. See §3.
2. **Hand-written Obsidian notes** — personal notes and class notes the user
   authors directly in one or more Obsidian vaults. Markdown, frequently
   edited, renamed, and deleted. See §4. **This is new and it is the biggest
   change to the original design.**
3. **Class materials** — docx, pptx, xlsx, pdf.

The goal is asking questions across a semester's worth of study material and
getting answers with citations back to the source — for videos, a deep link to
the exact timestamp; for vault notes, a link that opens the note.

The original brief assumed the RAG was a read-only consumer of a machine-written
S3 corpus. It is not. Corpus 2 is human-authored, mutable, and lives in GitHub
with no S3 copy. Everything in §4 follows from that.

## 2. Inherited constraints — DECIDED

- **Budget: under $10/month for both projects combined.** Video Vault costs
  ~$3–4. The realistic ceiling for the RAG is therefore ~$5/month. This is the
  single hardest constraint and it eliminates most published AWS RAG
  architectures — see §6.
- **This is a portfolio project.** Both projects exist partly to demonstrate AWS
  and AI engineering to recruiters. A demoable URL is worth real money here.
  Note the tension this now has with corpus 2 — see §7.
- **Corpus is small — smaller than the first estimate.** The original brief
  guessed 100k–200k chunks. Measured against real artifacts on 2026-08-02:

  | Measured on a real 18.9-minute video | Value |
  |---|---|
  | `summary.sections` | 11 |
  | `summary.takeaways` | 8 |
  | Summary JSON | 4,848 chars |
  | Transcript | 18,260 chars ≈ 4,565 tokens (521 caption segments) |

  At 80 videos/month that is roughly **12 summary chunks and 6–7 transcript
  chunks per video**, so **~18k–25k chunks in year one**, not 100k. At 1024
  dimensions that is **~80–100MB of vectors plus ~30MB of chunk text**. The
  100k–200k figure is a three-to-five-year ceiling, not a starting point.

  **Do not design for scale that will not arrive.** Re-verify these numbers
  against the bucket before sizing anything — the video mix may skew longer.

## 3. Video Vault interface contract — DECIDED, verified against shipped code

Both projects live in **one AWS account, region `us-east-2`**, so this is IAM
access, not cross-account.

Video Vault writes a self-contained artifact per video to S3. The bucket name is
published to SSM at **`/video-vault/content-bucket`** — read it with
`data "aws_ssm_parameter"` (Terraform) or an SDK call. Do not hardcode it, and do
not create a CloudFormation export dependency between the stacks.

Two prefixes, both confirmed populated in the live bucket:

| Key | Contents |
|---|---|
| `transcripts/{video_id}.json` | Raw transcript: `{video_id, language, segments:[{start_seconds, text}]}`. Full text, for recall on details the summary drops. |
| `summaries/{video_id}.json` | The artifact below. |

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "How Kubernetes Scheduling Actually Works",
  "channel": "Some Channel",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "published_at": "2026-07-01T12:00:00Z",
  "duration_seconds": 3862,
  "summarized_at": "2026-07-22",
  "note_path": "Video Vault/2026/How Kubernetes Scheduling Actually Works-dQw4w9WgXcQ.md",
  "summary": {
    "verdict": "...",
    "tldr": "...",
    "takeaways": ["..."],
    "sections": [
      {"start_seconds": 1120, "title": "Custom scheduler", "summary": "..."}
    ],
    "tags": ["kubernetes"]
  }
}
```

Three properties worth exploiting:

- **The artifact is self-contained by design.** It repeats title, channel, URL,
  and duration so an ingester processes one S3 object with no DynamoDB lookup.
- **`summary.sections` is a natural chunk boundary**, and each `start_seconds`
  lets a citation deep-link into the source video
  (`{url}&t={start_seconds}`). Most RAG demos cannot cite that precisely. Use it.
- **`note_path` is the join key to corpus 2.** It is the exact path of the
  markdown note in the vault repo. That is what makes the dedupe rule in §4
  mechanical rather than heuristic.

This corpus is **immutable**: an object is written once per video and only
rewritten if `scripts/resummarize.py` is run after a prompt change. A re-summarize
pass rewrites `summaries/` for the whole back catalogue in one go, so the ingester
must tolerate bulk updates — but it never has to handle deletes here.

## 4. The Obsidian vault as a corpus — NEW, mostly OPEN

The user wants to drop personal notes and class notes into the Obsidian vault and
ask questions about them. This corpus differs from corpus 1 on every axis that
matters to an ingester.

**Where it lives.** The vault is a **private GitHub repo**, separate from this
public one. Video Vault already writes to it via the GitHub contents API
(`src/vault_repo/committer.py`), authenticating with a token held in SSM as a
`SecureString`. Owner and repo name are CDK context values, not committed. The
same pattern is available to the RAG.

**GitHub is the source of truth, and there is no S3 copy.** The Video Vault spec
argued for the S3 write on the grounds that "reading them out of the private
GitHub repo would mean cloning or hammering the GitHub API." That argument holds
for video summaries and is unaffected. It does **not** rescue corpus 2 — nothing
writes hand-written notes to S3, so the RAG needs a GitHub read path regardless.
Deciding how that path works is the first real design question of this project.

**Dedupe rule — RECOMMENDED.** Video notes exist twice: as JSON under
`summaries/` in S3, and as rendered markdown under `Video Vault/{year}/` in the
vault. Ingesting both double-counts every video. Recommendation: **ingest videos
from S3** (structured, carries `start_seconds` for timestamp citations, no
markdown parsing) and **exclude the `Video Vault/` prefix from the vault
ingester**. The rendered markdown is strictly lossier than the JSON. Confirm with
the user, but this one has an obvious right answer.

**Mutability is the hard part.** Hand-written notes get edited, renamed, split,
merged, and deleted. A rename is a delete plus an add as far as any index is
concerned, and stale chunks from a deleted note are worse than a missing note
because they get cited confidently. Whatever the vector store turns out to be, it
needs upsert-by-path and delete-by-path, and the ingester needs a change-detection
mechanism — a stored last-seen commit SHA plus `git diff --name-status`, or
per-file content hashes.

**Full re-index is cheap enough to be a legitimate answer.** Titan Embed v2 is
~$0.02 per million tokens. The entire year-one corpus is roughly 5–6M tokens, so
**re-embedding everything from scratch costs about 11 cents.** Incremental
indexing is therefore a *latency and elegance* decision, not a cost decision.
Say that out loud in the interview — it kills a lot of premature complexity.

**Obsidian markdown has structure worth using, not stripping:**

- YAML frontmatter — free metadata filters. Video Vault's own notes carry
  `title`, `channel`, `url`, `video_id`, `duration`, `published`, `saved`,
  `summarized`, `tags`, `status`.
- `#tags` and frontmatter tags — the user's own topical labels.
- `[[wikilinks]]` — an explicit graph the user has already drawn between notes.
  Usable for retrieval expansion (pull in linked notes), or ignorable. Real
  option, no obvious answer.
- Headings — the natural chunk boundary for a hand-written note, the way
  `sections` is for a video.

**Multiple vaults.** The user said "vault(s)". Treat the vault identifier as a
first-class metadata field and filter dimension from the start; retrofitting it
means re-indexing. Cheap now (see the 11 cents above), annoying later.

**Citations.** Vault notes have no URL. Options: an `obsidian://open?vault=
{vault}&file={path}` URI, which opens the real note on the user's machine, or a
`github.com` blob link, which works anywhere including a demo. Probably both.
Video citations keep the timestamp deep link from §3 either way.

## 5. Build order — DECIDED, status updated 2026-08-02

Video Vault first, RAG second. Reasons: the RAG needs a corpus to be testable at
all; Video Vault has a clear done state while RAG scope is open-ended; and the
back catalogue grows while the RAG is being designed.

**Current status:** Video Vault deployed to `us-east-2` on 2026-08-01 and is
running end to end. As of 2026-08-02 the bucket holds **2 summaries and 2
transcripts** — enough to write and test an ingester against real data, nowhere
near enough to tune retrieval quality. At 80 videos/month there will be a few
hundred notes within a few months.

The practical consequence: **design and build now, but do not design an
evaluation that assumes a large corpus exists.** See the evaluation item in §10.

## 6. Cost analysis already done — DECIDED (the traps)

**Do not use OpenSearch Serverless.** It is the default in most Bedrock Knowledge
Bases tutorials and its minimum billable capacity runs to the **hundreds of dollars
per month while idle** — 50–100× the entire budget, for a corpus that would fit in
RAM. Any tutorial that starts there is written for enterprise scale.

**Aurora Serverless v2 with pgvector at a 0.5 ACU floor is ~$43/month** — also over
budget. A scale-to-zero configuration brings idle cost down to roughly storage,
which may be viable; verify current behavior and resume latency before committing.

Embeddings are a rounding error, and the measurements in §2 make that concrete:
the full year-one corpus is ~5–6M tokens, so a complete re-embed costs **~$0.11**.
**The vector store is the only real cost driver.** Size that decision first;
everything else is noise.

## 7. Hosting analysis already done — RECOMMENDED, not decided

The prior session recommended **AWS over local**, primarily because a local RAG is
nearly undemoable — there is no URL to send a recruiter, and the machine must be
on. Cost at this scale is roughly a wash (~$1–3/month either way).

The specific pattern suggested was: **build the index in batch, store it in S3,
load it into Lambda memory at query time.** The original sizing assumed ~800MB and
therefore a large Lambda. The measured numbers in §2 put year one at **~130MB
total (vectors plus chunk text)**, which fits in a 512MB–1GB Lambda and makes cold
starts markedly faster than the "few seconds" originally estimated. This makes the
pattern *more* attractive than when it was first proposed, and idle cost is
effectively zero.

**Confirm this with the user before building on it.** It is still a recommendation,
not a decision.

**Privacy — the flag from the original brief, now escalated.** The original
concern was narrow: class materials may be covered by course or institutional
policy on redistribution. Corpus 2 makes it broader, because *personal* notes are
now in scope. Two things need explicit answers:

1. **Storage.** Is the user comfortable with personal notes living in a cloud
   index? If not, there is a clean split — run ingestion locally and store only
   embeddings plus derived data in AWS, since vectors are not reversible to
   source text. Note this only works if chunk text is not also stored, which
   most retrieval designs need it to be. Do not hand-wave this.
2. **Exposure.** The portfolio argument in this section rests on a demoable URL.
   A demoable URL over a corpus of personal notes is a bad idea. The obvious
   resolution is auth on the real endpoint plus a public demo scoped to the video
   corpus only, but that is the user's call, and it makes auth a
   requirement rather than a nice-to-have. **This is a genuine conflict between
   two of the project's own goals — surface it early, do not resolve it
   silently.**

## 8. IaC — RECOMMENDED, not decided

Video Vault uses **AWS CDK (Python)**, chosen over SAM because CDK has real Lambda
bundling, typed Step Functions constructs, and in-memory infrastructure unit tests.

The prior session recommended **Terraform for this project** instead, so the
portfolio covers both tools, each used where it is genuinely the better fit —
Terraform suits a stack that is mostly managed data services rather than bundled
Lambdas. The interview value is in being able to explain *why* each was chosen.

Two things to check before committing:

- Terraform's AWS provider sometimes lags on newly released services. If this
  design depends on something recent, verify provider support.
- The §7 recommendation (build an index, load it into Lambda) is Lambda-bundling
  heavy, which is exactly the workload CDK was chosen for. If the design lands
  there, the "Terraform fits this better" argument weakens and should be
  re-examined rather than honoured out of inertia.

## 9. Bedrock reality on this account — DECIDED (learned the hard way)

The original brief said "Video Vault uses Sonnet 5 on Bedrock." **It does not.**
Corrected facts, all verified during the Video Vault build:

- The deployed summarizer runs **`us.anthropic.claude-sonnet-4-6`**, and the model
  ID is **configuration** (repo variable → CDK context → Lambda env), not a
  constant. Requests must set `thinking={"type": "disabled"}` and must **not** set
  `temperature`, `top_p`, or `top_k` — these models return 400 on non-default
  sampling parameters.
- **This account is not entitled to Sonnet 5**, despite
  `get-foundation-model-availability` reporting `AUTHORIZED`/`AVAILABLE` on all
  four fields. The inference endpoint returns *"anthropic.claude-sonnet-5 is not
  available for this account."* Do not plan around Sonnet 5 being available.
- **Use the classic `AnthropicBedrock` client, not `AnthropicBedrockMantle`.**
  Mantle serves only Sonnet 5 and is not served in `us-east-2` at all. The two
  clients need different IAM: classic wants `bedrock:InvokeModel`, Mantle wants
  `bedrock-mantle:CreateInference`.
- **A `us.`-prefixed model ID is a cross-region inference profile.** It needs
  `bedrock:InvokeModel` on the underlying foundation-model ARN in *every* member
  region (`us-east-1`, `us-east-2`, `us-west-2`), not just the profile ARN.
- **Bedrock's Anthropic use-case form is per-region and gates every Anthropic
  model.** Without it, every call returns 404 *"Model use case details have not
  been submitted."* It was submitted for `us-east-1` only. This is why the stack
  lives in `us-east-2` but the summarizer sets `BEDROCK_REGION=us-east-1`. **If
  the RAG's generation model is Anthropic, it inherits this — either call
  `us-east-1` or submit the form for `us-east-2` first.** One call can slip
  through before the gate applies, so never trust a single success.
- Embedding models are a different story: `amazon.titan-embed-text-v2:0` and
  `cohere.embed-v4:0` both list in `us-east-2`, and neither is Anthropic, so the
  use-case form does not apply. Listing is not entitlement — make one real call
  before designing around either.
- Secrets live only in SSM Parameter Store as `SecureString`. Never in env vars,
  never in code. The RAG's GitHub token follows the same rule.

## 10. OPEN — interview the user on these

Do not decide these unilaterally.

1. **Vector store.** S3-index-in-Lambda, Aurora pgvector scaled to zero, LanceDB,
   sqlite-vec, or something else. Driven by the budget in §2 and the traps in §6.
   Must support delete-by-path, per §4.
2. **How the vault gets read.** GitHub webhook into API Gateway, scheduled pull
   with a stored last-seen commit SHA, or a local push. Video Vault already polls
   on a schedule, so a pull is the consistent choice; a webhook is the more
   elegant one. Real tradeoff.
3. **Incremental index vs. full rebuild.** §4 shows a full rebuild costs ~11
   cents, so argue this on latency and complexity, not cost.
4. **Chunking strategy — now three shapes, not two.** Video sections are
   pre-chunked (§3). Hand-written markdown chunks on headings (§4). pptx, xlsx,
   and pdf each chunk differently, and xlsx may not belong in a text RAG at all.
5. **Whether transcripts get embedded, or only summaries.** Transcripts roughly
   double the chunk count (§2) for recall on details the summary drops. Real
   tradeoff, no obvious answer.
6. **One index or several.** Video notes, personal notes, class notes, class
   materials. Affects retrieval quality, filtering, and — given §7 — what a public
   demo is allowed to touch.
7. **Whether `[[wikilinks]]` are used for retrieval expansion** or treated as
   plain text.
8. **Query interface.** Web UI, CLI, Obsidian plugin, Slack, API only. This
   determines the demoability the whole hosting argument in §7 rests on, and per
   §7 it probably needs auth.
9. **Privacy posture for personal notes.** See §7. Answer before building, not
   after indexing them.
10. **Ingestion trigger for binary class files.** S3 upload, a watched folder,
    manual CLI. Note these are *not* markdown and probably do not belong in the
    Obsidian vault at all — worth asking whether the user intends to put them
    there or somewhere else.
11. **Evaluation.** How does the user judge whether retrieval is any good? Worth
    settling before building, not after. Constrained by §5: there are 2 videos
    today, so a golden-question set will have to be written by hand against a
    small corpus and grown over time.
12. **Generation model.** See §9 — the default is Sonnet 4.6 via `us-east-1`, not
    Sonnet 5. Confirm; a cheaper model may suffice for grounded answering, and
    grounded answering is the easiest place in the whole system to downgrade.

## 11. How the user prefers to work

Observed over the Video Vault design session and build:

- Asks for explicit tradeoff analysis and a recommendation — not a menu of options
  with no opinion. Lead with the recommendation, then justify it.
- Wants cost stated in real numbers, not adjectives. Free-tier limits and where
  they run out matter.
- Wants resume and portfolio impact called out explicitly when it affects a choice.
- Pushes back and asks follow-up questions on design decisions. Expect to defend
  reasoning; expect to change your answer when the priority shifts. (The SAM → CDK
  switch happened exactly this way.)
- Prefers free or near-free options, and wants the case made when paying is worth it.
- Works TDD with one task per commit, conventional commit prefixes, and lint and
  tests green before every commit. Infrastructure gets unit-tested too. Assume the
  RAG repo follows the same discipline unless told otherwise.
