data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "indexer" {
  name               = "notes-rag-indexer"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "indexer_logs" {
  role       = aws_iam_role.indexer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "indexer" {
  # Read-only on the Video Vault bucket, and only the prefixes the handler
  # actually reads (var.source_prefixes) - not the whole bucket. GetObject and
  # ListBucket need different resource shapes (object ARNs vs. the bucket
  # ARN), so they are split into two statements rather than one broad grant.
  statement {
    sid     = "ReadSourceBucket"
    actions = ["s3:GetObject"]
    resources = [
      for prefix in var.source_prefixes : "arn:aws:s3:::${var.source_bucket}/${prefix}*"
    ]
  }

  statement {
    sid       = "ListSourceBucket"
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.source_bucket}"]

    # Without this, ListBucket would still be scoped to the bucket as a
    # whole - the resource ARN for ListBucket can only ever be the bucket
    # itself, never an object path. The s3:prefix condition is what actually
    # confines the *listing* to the watched prefixes.
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = [for prefix in var.source_prefixes : "${prefix}*"]
    }
  }

  statement {
    sid = "ReadWriteIndexBucket"
    # s3:ListBucket is not here for listing - the handler never lists this
    # bucket. It is here because, without it, S3 returns 403 AccessDenied
    # instead of 404 NoSuchKey for a GetObject on a key that doesn't exist,
    # to avoid revealing whether the key is present. That masking breaks the
    # first-run path: on a brand-new index bucket neither index/manifest.json
    # nor index/full.db exists yet, and both get_json and download_file rely
    # on distinguishing "absent" (proceed as first run) from "denied" (real
    # error). Do not remove this again on a "the handler never lists this
    # bucket" argument - that argument is correct and irrelevant.
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.index.arn,
      "${aws_s3_bucket.index.arn}/*",
    ]
  }

  # Scoped to the one embedding model, not bedrock:*.
  statement {
    sid     = "InvokeTitanEmbeddings"
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:${var.region}::foundation-model/${var.embed_model_id}",
    ]
  }
}

resource "aws_iam_role_policy" "indexer" {
  name   = "notes-rag-indexer"
  role   = aws_iam_role.indexer.id
  policy = data.aws_iam_policy_document.indexer.json
}
