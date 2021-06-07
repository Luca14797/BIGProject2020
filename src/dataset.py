from boto3 import Session
from botocore.exceptions import ClientError

import os


def main():

    # MACRO
    BUCKET_NAME = "s3-terraform-bucket-big"
    ACCESS_KEY = ""     # Insert your ACCESS KEY
    SECRET_KEY = ""     # Insert your SECRET KEY
    TOKEN = ""          # Insert your TOKEN

    # Create folder dataset
    if not os.path.exists("../dataset"):
        os.makedirs("../dataset")

    # Create folder split in folder dataset
    if not os.path.exists("../dataset/split"):
        os.makedirs("../dataset/split")

    # Create a session for access to AWS S3
    session = Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, aws_session_token=TOKEN)

    # Create S3 resource
    s3 = session.resource("s3")

    try:
        # Download the files contained in S3 bucket and save this file in dataset folder
        s3.Bucket(BUCKET_NAME).download_file("info_texts.json", "../dataset/info_texts.json")
        s3.Bucket(BUCKET_NAME).download_file("test.txt", "../dataset/split/test.txt")
        s3.Bucket(BUCKET_NAME).download_file("train.txt", "../dataset/split/train.txt")
        s3.Bucket(BUCKET_NAME).download_file("val.txt", "../dataset/split/val.txt")
        s3.Bucket(BUCKET_NAME).download_file("texts_list.txt", "../dataset/texts_list.txt")

    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist")
        else:
            raise


if __name__ == '__main__':
    main()
