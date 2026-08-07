#!/usr/bin/env bash
set -euo pipefail

# Sync one Obsidian vault into the indexer's source bucket.
#
# --delete is not optional. The indexer's manifest turns a removed object into
# a chunk deletion, so without it a note deleted in Obsidian stays answerable
# forever - the worst kind of stale, because the answer still cites a file the
# user believes they destroyed.
#
# Only *.md is uploaded. The indexer would skip everything else by suffix
# anyway, but not uploading it is cheaper, keeps .obsidian workspace state out
# of a bucket the Lambda can read, and removes an entire class of oversized-
# object failure before it can reach the index.

usage() {
  echo "usage: NOTES_BUCKET=<bucket> $0 <vault-dir> <vault-id> [aws s3 sync flags...]" >&2
  echo "example: NOTES_BUCKET=notes-rag-source-123 $0 ~/vaults/josh josh --dryrun" >&2
  exit 2
}

[ $# -ge 2 ] || usage

VAULT_DIR="$1"
VAULT_ID="$2"
shift 2

: "${NOTES_BUCKET:?set NOTES_BUCKET to the indexer source bucket}"
AWS_REGION="${AWS_REGION:-us-east-2}"

[ -d "$VAULT_DIR" ] || { echo "no such directory: $VAULT_DIR" >&2; exit 1; }

case "$VAULT_ID" in
  */*|"")
    # vault_id becomes an S3 prefix segment and a chunk's vault_id, which is
    # embedded in every content_hash. A slash would silently split the prefix.
    echo "vault-id must be a single path segment, got: '$VAULT_ID'" >&2
    exit 1
    ;;
esac

echo "syncing $VAULT_DIR -> s3://$NOTES_BUCKET/notes/$VAULT_ID/"

aws s3 sync "$VAULT_DIR" "s3://$NOTES_BUCKET/notes/$VAULT_ID/" \
  --region "$AWS_REGION" \
  --delete \
  --exclude '*' \
  --include '*.md' \
  --exclude '.obsidian/*' \
  --exclude '.trash/*' \
  "$@"

echo "done. the indexer picks this up within 5 minutes, or invoke it now:"
echo "  aws lambda invoke --function-name notes-rag-indexer --region $AWS_REGION \\"
echo "    --cli-binary-format raw-in-base64-out --payload '{}' /dev/stdout"
