output "index_bucket" {
  value = aws_s3_bucket.index.id
}

output "indexer_function_name" {
  value = aws_lambda_function.indexer.function_name
}

output "indexer_role_arn" {
  value = aws_iam_role.indexer.arn
}
