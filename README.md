# Notes RAG

Ask questions across a semester of study material and get answers with citations
back to the source: for videos, a deep link to the exact timestamp.

Chunkers turn Video Vault summaries and transcripts into `Chunk`s; a shared
normalizer merges, splits, context-prefixes and hashes them; vectors and metadata
live together in one `sqlite-vec` file. An indexer Lambda rebuilds that file on a
schedule, re-embedding only the chunks whose content actually changed.

## Local use

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -e '.[dev]'

# Build an index from a directory of Video Vault artifacts
.venv/bin/notes-rag-index ./artifacts --out index.db --vault-id "Class Notes"

# Score retrieval against the golden question set
.venv/bin/python -m eval.run --index index.db --questions eval/questions.yaml --k 6
```

Add `--fake-embedder` to either command to run with the deterministic embedder
and no AWS credentials.

## Tests

```bash
.venv/bin/pytest                  # unit tests; no credentials needed
.venv/bin/pytest -m integration   # touches AWS; needs credentials in us-east-2
```

## Deployment

Requires Terraform >= 1.10 and AWS credentials for `us-east-2`.

```bash
# 1. Once per account: create the Terraform state bucket.
cd infra/bootstrap
terraform init
terraform apply -var="state_bucket=notes-rag-tfstate-<account-id>"
cd ../..

# 2. Build the Lambda bundle. Re-run this before ANY apply that should pick up
#    code changes: Terraform derives source_code_hash from the zip, so skipping
#    it deploys the previous code with no error and no diff.
./scripts/build_lambda.sh

# 3. Deploy.
cd infra
terraform init -backend-config="bucket=notes-rag-tfstate-<account-id>"
terraform apply -var="index_bucket=notes-rag-index-<account-id>"
```

The indexer then runs every 5 minutes. Trigger one immediately with:

```bash
aws lambda invoke --function-name notes-rag-indexer --region us-east-2 \
  --cli-binary-format raw-in-base64-out --payload '{}' /dev/stdout
```

### Why the bundle ships its own SQLite

AWS Lambda's managed `python3.12` runtime builds the stdlib `sqlite3` **without**
loadable-extension support, so `sqlite-vec` cannot load through it. The bundle
includes `pysqlite3-binary`, which carries its own SQLite with extensions
enabled, and `src/notes_rag/store/sqlite_vec.py` prefers it when present. Dropping
that dependency breaks the indexer on its first database connection.

### Why the Lambda has no reserved concurrency

`infra/indexer.tf` does not set `reserved_concurrent_executions`. It did
originally, at `1`, to guarantee two runs never race on the index artifact - but
that requires an account-level Lambda concurrency quota of at least 11 (AWS
refuses any reservation that would drop unreserved capacity below 10), and a
freshly provisioned account's default quota is exactly 10. No reservation, of
any size, is possible there.

The same guarantee now comes from `timeout = 240` being shorter than the
5-minute schedule interval: a scheduled run always finishes or is killed before
the next one fires, so two scheduled runs can never overlap, and
`maximum_retry_attempts = 0` on the schedule target means nothing queues up
behind a failure either. The accepted residual risk is narrower: an ad hoc
`aws lambda invoke` can still land on top of an in-flight scheduled run, with a
lost update that self-heals on the next tick as the worst case.
