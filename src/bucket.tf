provider "aws" {
    region = "us-east-1"
    access_key = "<YOUR ACCESS KEY>"
    secret_key = "<YOUR SECRET KEY>"
    token = "<YOUR TOKEN>"
}

# Create a bucket
resource "aws_s3_bucket" "bucket_tweets" {
    bucket = "s3-terraform-bucket-big"

    acl = "public-read"

    tags = {
        Name = "S3 bucket tweets"

        Environment = "Dev"

    }

}

# Upload data in bucket
resource "aws_s3_bucket_object" "object_1" {
    for_each = fileset("BIGProject2020/dataset/", "*")

    bucket = aws_s3_bucket.bucket_tweets.id

    key = each.value

    source = "BIGProject2020/dataset/${each.value}"

    etag = filemd5("BIGProject2020/dataset/${each.value}")

}

# Upload data in bucket
resource "aws_s3_bucket_object" "object_2" {

    for_each = fileset("BIGProject2020/dataset/splits", "*")

    bucket = aws_s3_bucket.bucket_tweets.id

    key = each.value

    source = "BIGProject2020/dataset/splits/${each.value}"

    etag = filemd5("BIGProject2020/dataset/splits/${each.value}")

}
