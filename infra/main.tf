terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Native S3 state locking. No DynamoDB table: that requirement is obsolete as
  # of Terraform 1.10. Bucket name is passed with -backend-config at init time,
  # because it is created by the bootstrap stack.
  backend "s3" {
    key          = "notes-rag/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "notes-rag"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
