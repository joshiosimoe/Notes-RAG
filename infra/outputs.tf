output "index_bucket" {
  value = aws_s3_bucket.index.id
}

output "indexer_function_name" {
  value = aws_lambda_function.indexer.function_name
}

output "indexer_role_arn" {
  value = aws_iam_role.indexer.arn
}

output "indexer_alarm_topic_arn" {
  value = aws_sns_topic.indexer_alarms.arn
}

output "notes_bucket" {
  value = aws_s3_bucket.source.id
}
