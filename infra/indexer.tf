resource "aws_lambda_function" "indexer" {
  function_name = "notes-rag-indexer"
  role          = aws_iam_role.indexer.arn
  handler       = "notes_rag.indexer.handler.lambda_handler"
  runtime       = "python3.12"
  architectures = ["x86_64"]

  filename         = var.lambda_zip
  source_code_hash = filebase64sha256(var.lambda_zip)

  # 240 < the 300s schedule interval below, on purpose: a scheduled run always
  # finishes or is killed before the next one fires, so two scheduled runs can
  # never overlap. Do not raise this without also raising schedule_expression -
  # the two are coupled.
  timeout     = 240
  memory_size = 1024

  # /tmp holds full.db plus public.db. The spec sizes full.db at ~130MB at 20k
  # chunks; the 512MB default is thinner headroom than a 5-minute job deserves.
  ephemeral_storage {
    size = 2048
  }

  # No reserved_concurrent_executions here: this account's Lambda concurrency
  # quota is 10, and AWS refuses any reservation that would drop unreserved
  # capacity below 10 - so no reservation, of any size, is possible on this
  # account. The no-overlap guarantee this was meant to provide now comes from
  # timeout (240s) being shorter than the schedule interval (300s) instead.
  # Residual risk, accepted: an ad hoc `aws lambda invoke` can still land on
  # top of an in-flight scheduled run - worst case is a lost update that
  # self-heals on the next tick.

  environment {
    variables = {
      SOURCE_BUCKET    = var.source_bucket
      SOURCE_PREFIXES  = "summaries/,transcripts/"
      INDEX_BUCKET     = aws_s3_bucket.index.id
      EMBED_DIMENSIONS = tostring(var.embed_dimensions)
      BEDROCK_REGION   = var.region
    }
  }
}

resource "aws_cloudwatch_log_group" "indexer" {
  name              = "/aws/lambda/${aws_lambda_function.indexer.function_name}"
  retention_in_days = 14
}

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "notes-rag-indexer-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

data "aws_iam_policy_document" "scheduler_invoke" {
  statement {
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.indexer.arn]
  }
}

resource "aws_iam_role_policy" "scheduler_invoke" {
  name   = "invoke-indexer"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler_invoke.json
}

resource "aws_scheduler_schedule" "indexer" {
  name = "notes-rag-indexer"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = "UTC"

  target {
    arn      = aws_lambda_function.indexer.arn
    role_arn = aws_iam_role.scheduler.arn

    # The handler ignores the event; an empty payload keeps the scheduled path
    # and an on-demand `aws lambda invoke` identical.
    input = jsonencode({})

    retry_policy {
      # The next tick is 5 minutes away and does the same work, so retrying a
      # failed run buys nothing.
      maximum_retry_attempts = 0
    }
  }
}
