# A failing indexer has no other symptom. maximum_retry_attempts = 0 and there
# is no DLQ, so a run that raises is dropped on the floor: the manifest never
# advances, full.db keeps serving whatever it held at the last good run, and
# every query still answers - just from a corpus that stopped growing. Nothing
# in the system says so. This alarm is the only thing that does.

resource "aws_sns_topic" "indexer_alarms" {
  name = "notes-rag-indexer-alarms"

  # No kms_master_key_id: the AWS-managed alias/aws/sns key cannot grant
  # cloudwatch.amazonaws.com the kms:GenerateDataKey it needs to publish, and
  # its key policy is not editable. Encrypting this topic would mean a
  # customer-managed key for payloads that carry nothing but an alarm name.
}

resource "aws_sns_topic_subscription" "indexer_alarms_email" {
  # Opt-in: with no alarm_email the alarm still fires and is still visible in
  # the console and via the CLI, it just doesn't page anyone. Creating a
  # subscription sends a confirmation email that must be clicked before any
  # notification is delivered - until then this resource exists in
  # PendingConfirmation and nothing arrives.
  count = var.alarm_email == "" ? 0 : 1

  topic_arn = aws_sns_topic.indexer_alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

resource "aws_cloudwatch_metric_alarm" "indexer_errors" {
  alarm_name  = "notes-rag-indexer-errors"
  namespace   = "AWS/Lambda"
  metric_name = "Errors"
  dimensions = {
    FunctionName = aws_lambda_function.indexer.function_name
  }

  statistic           = "Sum"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1

  # One period per scheduled tick, and two consecutive breaching ticks before
  # anyone is told. A single failed run is not worth a page: the next tick is
  # five minutes away and does exactly the same work, so a transient Bedrock
  # throttle or a lost S3 read self-heals before this alarm would have fired.
  # Two in a row does not self-heal - that is a wedge.
  period              = 300
  evaluation_periods  = 2
  datapoints_to_alarm = 2

  # Lambda publishes Errors = 0 on a successful invocation, so a healthy
  # 5-minute schedule leaves no gaps here. Missing data therefore means the
  # function was not invoked at all - a disabled or broken schedule, which
  # goes stale exactly as silently as a raising handler does. Treating it as
  # breaching covers both failure modes with one alarm. The cost: deliberately
  # disabling the schedule will page too.
  treat_missing_data = "breaching"

  alarm_description = "notes-rag-indexer failed twice in a row, or stopped being invoked. The index is going stale; check /aws/lambda/notes-rag-indexer."
  alarm_actions     = [aws_sns_topic.indexer_alarms.arn]
  ok_actions        = [aws_sns_topic.indexer_alarms.arn]
}
