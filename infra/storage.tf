# Holds full.db, public.db, and the ETag manifest. Private: the demo path in a
# later plan reads public.db through a Lambda, never directly from the browser.
resource "aws_s3_bucket" "index" {
  bucket = var.index_bucket
}

resource "aws_s3_bucket_public_access_block" "index" {
  bucket                  = aws_s3_bucket.index.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "index" {
  bucket = aws_s3_bucket.index.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# One noncurrent version is enough to roll back a bad index build; keeping more
# would grow storage without adding recovery value.
resource "aws_s3_bucket_versioning" "index" {
  bucket = aws_s3_bucket.index.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "index" {
  bucket = aws_s3_bucket.index.id

  rule {
    id     = "expire-old-index-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      newer_noncurrent_versions = 1
      # 1 is the minimum AWS accepts alongside newer_noncurrent_versions. This
      # rule keeps the single newest noncurrent version (for rollback) and
      # expires every other noncurrent version once it's been noncurrent for
      # a day. At 7 here, a version was retained for a week after it stopped
      # being the newest noncurrent one - on a schedule producing one artifact
      # version per 5-minute tick, that's roughly a week of history, not one
      # version, contradicting the comment above.
      noncurrent_days = 1
    }
  }
}
