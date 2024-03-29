# Big Data Project

1. [Introduction](#Introduction)
2. [Project description](#Project-description)
3. [Implementation](#Implementation)
4. [Run Project](#Run-project)
5. [Results](#Results)
3. [Contributors](#Contributors)

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

The goal of this project is to determine if the text of a tweet contains elements related to hatred, whether it is racial, sexist, etc. or not.
The classification problem is solved through the use of a Bag-of-Words; the extracted features are then used for the training of a Multilayer Perceptron.

To speed up the execution of the project, the Apache Spark framework was used, which allows you to distribute the work among the nodes of a cluster.

The tweets are contained in a json file and each of them is described by the following fields:
* **id**: unique tweet identifier
* **img_url**: url of the image, if it is present
* **labels**: list of 3 labels, as a number
* **tweet_url**: url of tweet
* **tweet_text**: text of the tweet
* **labels_str**: list of 3 labels, as a string

The dataset is saved inside a Dataframe which is partitioned according to the number of cores available in the cluster 
according to the following formula: ```numClusterCores * replicationFactor```.
This is done to reduce execution times as each worker in the cluster works on a subset of the original dataset. 
For example, if the cluster contains 3 slave nodes with 2 cores for each node and the replication factor is 2, 
the number of partitions of the Dataframe is 12.

The Bag-of-Words is created by first dividing the texts into single words and then calculating the frequency of the terms,
i.e. the number of times a word appears in the text.

After creating the Bag-of-Words, a Multilayer Perceptron was trained using Apache Spark's MLlib library.

## Run project

1. [Download and Install](https://learn.hashicorp.com/tutorials/terraform/install-cli?in=terraform/aws-get-started) Terraform.


2. Clone [Terraform project](https://github.com/martinasalis/Terraform_project).
```bash
git clone https://github.com/martinasalis/Terraform_project.git
```

3. Enter in the Terraform project directory.
```bash
cd Terraform_project/
```

4. Login in your AWS account and create a key pairs in **PEM** format.
   Follow [this](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair) guide.
   After you create a key pairs download and save it in ```Terraform_project/``` folder. 
   Change the permission of the key pairs by inserting this command:
```bash
chmod 400 <YOUR KEY NAME>.pem
```
   

5. Open the file ```terraform.tfvars``` and insert your data.
    * If you are using AWS Educate you can retrieve your values in the Vocareum page you get after having logged in by clicking on the button "Account Details" under the voice "AWS CLI".
    * If you are using the normal AWS follow the guide on [this](https://aws.amazon.com/it/blogs/security/how-to-find-update-access-keys-password-mfa-aws-management-console/) page in the paragraph called "Generate access keys for programmatic access".
    
    The maximum number of worker nodes is 8.
```
access_key="<YOUR ACCESS KEY>"
secret_key="<YOUR SECRET KEY>"
token="<YOUR TOKEN>"
aws_private_key_name="<YOUR KEY NAME>"
aws_private_key_path="<YOUR KEY NAME>.pem"
slaves_count=<NUMBER CLUSTER WORKER>
```

6. Open terminal in ```Terraform_project/``` directory and insert these commands (one by one):
```bash
terraform init
terraform apply
```

7. After the cluster is created, the **PUBLIC IP** and **PUBLIC DNS** of the master node are shown.
   Connect to it using this command:
```bash
ssh -i '<YOUR KEY NAME>.pem' ubuntu@<PUBLIC DNS>
```

8. Start Hadoop and Spark cluster by inserting these commands (one by one) in the master node:
```bash
hdfs namenode -format
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
$HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-slaves.sh spark://namenode:7077
```

9. Download this project from GitHub in the master node.
```bash
git clone https://github.com/Luca14797/BIGProject2020.git
```

10. Execute the script ```dataset.sh``` to download the dataset from AWS S3. Run these commands (one by one):
```
chmod +x BIGProject2020/src/dataset.sh
sudo /bin/bash BIGProject2020/src/dataset.sh
```

11. Move the dataset of the project into HDFS for run correctly the project.
```bash
hdfs dfs -mkdir -p /user/ubuntu
hdfs dfs -put /home/ubuntu/BIGProject2020/dataset/ /user/ubuntu
```

12. During execution of the project you can control it on the Spark GUI on your browser. 
    Connect to ```<PUBLIC IP>:8080```.
    

13. Run the project.
```bash
cd BIGProject2020/src/
$SPARK_HOME/bin/spark-submit --master spark://namenode:7077 main.py
```

14. After the execution is finished, exit from master node.


15. Destroy the cluster using this command:
```bash
terraform destroy
```

## Results
The project was tested using 1 to 8 workers.

The first test consists in training a Multilayer Perceptron without the use of Cross-Validation and using instances on AWS of type ```t2.large```. The following table shows the results, time is reported in seconds.

|   # workers    |   Time (s)    |
|---    |---    |
|   1   |   228 |
|   2   |   132 |
|   3   |   102 |
|   4   |   84  |
|   5   |   72  |
|   6   |   66  |
|   7   |   60  |
|   8   |   55  |

The second test consists in training a Multilayer Perceptron using Cross-Validation and using instances on AWS of the type ```t2.medium```. The following table shows the results, time is reported in minutes.

|   # workers    |   Time (m)    |
|---    |---    |
|   1   |   19 |
|   2   |   11 |
|   3   |   7,9 |
|   4   |   6,2  |
|   5   |   5,2  |
|   6   |   4,7  |
|   7   |   4,5  |
|   8   |   4,2  |

## Contributors
[Martina Salis](https://github.com/martinasalis) <br/>
[Luca Grassi](https://github.com/Luca14797)
