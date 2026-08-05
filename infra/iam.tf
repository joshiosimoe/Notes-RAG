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
  # Read-only on the Video Vault bucket. The indexer must never be able to
  # modify the corpus it is indexing.
  statement {
    sid     = "ReadSourceBucket"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.source_bucket}",
      "arn:aws:s3:::${var.source_bucket}/*",
    ]
  }

  statement {
    sid     = "ReadWriteIndexBucket"
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
