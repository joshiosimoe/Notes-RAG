# Receives the Obsidian vault via scripts/sync_vault.sh. Deliberately not the
# index bucket: the indexer holds PutObject there, and a source its own
# consumer can overwrite is not a source.
resource "aws_s3_bucket" "source" {
  bucket = var.notes_bucket
}

resource "aws_s3_bucket_public_access_block" "source" {
  bucket                  = aws_s3_bucket.source.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "source" {
  bucket = aws_s3_bucket.source.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Versioning is the undo for `sync_vault.sh --delete`: a mistyped vault
# directory deletes every object in the prefix, and without versions that is
# unrecoverable.
resource "aws_s3_bucket_versioning" "source" {
  bucket = aws_s3_bucket.source.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "source" {
  bucket = aws_s3_bucket.source.id

  rule {
    id     = "expire-old-note-versions"
    status = "Enabled"

    filter {}

    # 30 days rather than the index bucket's 1: notes are kilobytes and this is
    # the only copy outside OneDrive, so retention is worth more here than the
    # storage it costs.
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# One list feeds both the Lambda's SOURCE_LIST and the IAM grant below, so the
# two cannot drift - the failure 07b819a had to fix.
locals {
  all_sources = concat(
    var.external_sources,
    [
      for vault in var.vaults : {
        bucket   = aws_s3_bucket.source.id
        prefixes = ["notes/${vault}/"]
        vault_id = vault
      }
    ],
  )
}
