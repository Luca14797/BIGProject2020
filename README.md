# BigData Project

1. [Introduction](#Introduction)
2. [Project description](#Project-description)
3. [Run Project](#Run-project)

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
* attacks to other communities.

## Run project

### Download project

Download the project from github.
```bash
git clone https://github.com/Luca14797/BIGProject2020.git
```

### Prepare dataset

Move the dataset of the project into HDFS for run correctly the project.
```bash
hdfs dfs -mkdir -p /user/ubuntu
hdfs dfs -put /home/ubuntu/BIGProject2020/dataset/ /user/ubuntu
```

### Execute project

The number beside ```main.py``` indicates the number of nodes created on aws to run the project.

**Depending on the number of active workers, the parameter value must be changed.**

In the command below, 8 was entered as a value because 8 worker nodes were used to run the project.
The maximum number of workers accepted is 8.
```bash
cd BIGProject2020/src/
$SPARK_HOME/bin/spark-submit --master spark://namenode:7077 main.py 8
```
