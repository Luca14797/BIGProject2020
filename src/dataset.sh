#!/bin/bash

sudo mkdir /home/ubuntu/BIGProject2020/dataset
sudo mkdir /home/ubuntu/BIGProject2020/dataset/split

wget https://s3-terraform-bucket-big.s3.amazonaws.com/train.txt -P /home/ubuntu/BIGProject2020/dataset/split
wget https://s3-terraform-bucket-big.s3.amazonaws.com/test.txt -P /home/ubuntu/BIGProject2020/dataset/split
wget https://s3-terraform-bucket-big.s3.amazonaws.com/val.txt -P /home/ubuntu/BIGProject2020/dataset/split
wget https://s3-terraform-bucket-big.s3.amazonaws.com/info_texts.json -P /home/ubuntu/BIGProject2020/dataset
wget https://s3-terraform-bucket-big.s3.amazonaws.com/texts_list.txt -P /home/ubuntu/BIGProject2020/dataset
