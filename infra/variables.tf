variable "region" {
  type    = string
  default = "us-east-2"
}

variable "index_bucket" {
  type        = string
  description = "Globally unique name for the bucket holding full.db and public.db."
}

variable "source_bucket" {
  type        = string
  description = "Video Vault content bucket holding summaries/ and transcripts/."
  default     = "videovaultstack-contentbucket52d4b12c-s0o3jpdq69b4"
}

variable "embed_model_id" {
  type    = string
  default = "amazon.titan-embed-text-v2:0"
}

variable "embed_dimensions" {
  type    = number
  default = 1024
}

variable "schedule_expression" {
  type        = string
  description = "EventBridge Scheduler expression for the indexer."
  default     = "rate(5 minutes)"
}

variable "lambda_zip" {
  type        = string
  description = "Path to the deployment package built by scripts/build_lambda.sh."
  default     = "../build/lambda.zip"
}
