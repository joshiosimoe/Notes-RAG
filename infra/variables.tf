variable "region" {
  type    = string
  default = "us-east-2"
}

variable "index_bucket" {
  type        = string
  description = "Globally unique name for the bucket holding full.db and public.db."
}

variable "notes_bucket" {
  type        = string
  description = "Globally unique name for the bucket the Obsidian vault syncs into. Separate from index_bucket on purpose: the indexer has write access there, and a source its own consumer can overwrite is not a source."
}

variable "vaults" {
  type        = list(string)
  description = "Vault ids hosted in notes_bucket under notes/<id>/. Each id becomes the chunk's vault_id and therefore part of every note chunk's content_hash, so renaming one re-embeds that vault's whole corpus."
  default     = ["joshiosimoe"]
}

variable "external_sources" {
  type = list(object({
    bucket   = string
    prefixes = list(string)
    vault_id = optional(string)
  }))
  description = "Source buckets this stack does not own. Prefixes must end in \"/\" - they are interpolated as \"<prefix>*\" into the IAM s3:prefix condition, so \"notes\" would also grant \"notes-private/\"."
  default = [
    {
      bucket   = "videovaultstack-contentbucket52d4b12c-s0o3jpdq69b4"
      prefixes = ["summaries/", "transcripts/"]
    },
  ]
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

variable "alarm_email" {
  type        = string
  description = "Address subscribed to the indexer alarm topic. Empty (the default) creates no subscription: the alarm still fires and is visible in CloudWatch, it just notifies nobody. A non-empty value sends a confirmation email that must be clicked before anything is delivered."
  default     = ""
}

variable "lambda_zip" {
  type        = string
  description = "Path to the deployment package built by scripts/build_lambda.sh."
  default     = "../build/lambda.zip"
}
