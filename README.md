# Big Data Project

1. [Introduction](#Introduction)
2. [Project description](#Project-description)
3. [Implementation](#Implementation)
4. [Run Project](#Run-project)
5. [Results](#Results)

## Introduction

Modern social media content usually include images and text. Some of these multimodal publications are only hate speech because of the combination of the text with a certain image. 

That is because the presence of offensive terms does not itself signify hate speech, and the presence of hate speech is often determined by the context of a publication. Moreover, users authoring hate speech tend to intentionally construct publications where the text is not enough to determine they are hate speech. 

This happens especially in Twitter, where multimodal tweets are formed by an image and a short text, which in many cases is not enough to judge them.

## Project description

This project implements the classification of text tweets using Spark and the python API (pySpark).

As will be described below, three different labels are associated with each tweet. A new one is generated from these labels.

To simplify the classification, it was decided to carry out a binary classification by associating one of the following classes to the tweets: 
* hate
* not hate

### Dataset

The dataset is composed of textual tweets which are associated with three labels that indicate the classification of the tweet. Each classification was done by three different people using the Amazon Mechanical Turk service. These people classified the tweets into six different categories: 
* no attacks to any community
* racist
* sexist
* homophobic
* religion based attacks 
* attacks to other communities

## Implementation

## Run project

1. [Download and Install](https://learn.hashicorp.com/tutorials/terraform/install-cli?in=terraform/aws-get-started) Terraform


2. [Download](https://github.com/martinasalis/Terraform_project) Terraform project


3. Enter in the Terraform project directory
```bash
cd Terraform_project/
```

4. Login in your AWS account and create a key pairs in **PEM** format.
   Follow [this](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair) guide.
   After you create a key pairs download and save it in ```Terraform_project/``` folder.
   

5. Open the file ```terraform.tfvars``` and insert your data.
    * If you are using AWS Educate you can retrive your values in the Vocareum page you get after having logged in by clicking on the button "Account Details" under the voice "AWS CLI".
    * If you are using the normal AWS follow the guide on [this](https://aws.amazon.com/it/blogs/security/how-to-find-update-access-keys-password-mfa-aws-management-console/) page in the paragraph called "Generate access keys for programmatic access".
```
access_key="<YOUR ACCESS KEY>"
secret_key="<YOUR SECRET KEY>"
token="<YOUR TOKEN>"
aws_private_key_name="<YOUR KEY NAME>"
aws_private_key_path="<YOUR KEY NAME>.pem"
slaves_count=<NUMBER CLUSTER WORKER>
```

6. Open terminal in ```Terraform_project/``` directory and insert these commands:
```bash
terraform init
terraform apply
```

7. After the cluster is created, the **PUBLIC IP** and **PUBLIC DNS** of the master node are shown.
   Connect to it using this command:
```bash
ssh -i '<YOUR KEY NAME>.pem' ubuntu@<PUBLIC DNS>
```

8. Download this project from github in master node.
```bash
git clone https://github.com/Luca14797/BIGProject2020.git
```

9. Move the dataset of the project into HDFS for run correctly the project.
```bash
hdfs dfs -mkdir -p /user/ubuntu
hdfs dfs -put /home/ubuntu/BIGProject2020/dataset/ /user/ubuntu
```

10. During execution of the project you can control it on the Spark GUI on your browser. 
    Connect to ```<PUBLIC IP>:8080```.
    

11. Run the project. The number beside ```main.py``` indicates the number of workers created on AWS to run the project.
    **Depending on the number of active workers, the parameter value must be changed.**
    In the command below, 8 was entered as a value because 8 workers were used to run the project. 
    The maximum number of workers accepted is 8.
```bash
cd BIGProject2020/src/
$SPARK_HOME/bin/spark-submit --master spark://namenode:7077 main.py 8
```

12. After the execution is finished, exit from master node and destroy the cluster using this command:
```bash
terraform destroy
```

## Results
The project was tested using 1 to 8 workers. The times obtained are as follows:

|   Num. workers    |   Time (s)    |
|---    |---    |
|   1   |   228 |
|   2   |   132 |
|   3   |   102 |
|   4   |   84  |
|   5   |   72  |
|   6   |   66  |
|   7   |   60  |
|   8   |   55  |
