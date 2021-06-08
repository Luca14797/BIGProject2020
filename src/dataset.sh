#!/bin/bash

mkdir ../dataset
mkdir ../dataset/split

wget https://s3-terraform-bucket-big.s3.amazonaws.com/train.txt -P ../dataset/split
wget https://s3-terraform-bucket-big.s3.amazonaws.com/test.txt -P ../dataset/split
wget https://s3-terraform-bucket-big.s3.amazonaws.com/val.txt -P ../dataset/split
wget https://s3-terraform-bucket-big.s3.amazonaws.com/info_texts.json -P ../dataset
wget https://s3-terraform-bucket-big.s3.amazonaws.com/texts_list.txt -P ../dataset
