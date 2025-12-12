---
layout: post
title: Run your own Apache Spark jobs in AWS EMR and S3
description: 
image: 
category: [Data Engineering]
tags: aws datalake pipeline
date: 2021-02-01 00:00 +0000
---
## Run your own Apache Spark jobs in AWS EMR and S3

Recently, I participated in Udacity's Nanodegree Program for Data Engineers. It's kinda like to review what I did past and refresh some tech stacks' knowledge for me in the Data Engineering field. Today, I'm gonna writing this article to avoid you spending extra efforts to run a Spark job or other Hadoop eco-systems jobs on AWS. 

This idea comes from the Data Lake demo below, you can find the part I tried to apply through Shell script and Makefile between [S3->EMR->S3] in the figure. The PySpark script I used reference from the project in Course Data Lake, click the photo to check detail.

[![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/datalake_demo.png){:class="img-responsive"}](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/datalake_demo.png)

Before we start, make sure you've already done with registering an AWS account and downloading the credential on [AWS console](https://aws.amazon.com/)(the easiest way, you bet).

### EMR workflow
First off, let us have a look at the EMR workflow, click the photo to check detail.

[![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/emr-workflow.png){:class="img-responsive"}](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/emr-workflow.png)

What we want to do, is to make sure the work of setting up and launching a cluster, adding working steps, creating log files, and terminating the cluster as simple as we can. AWS CLI(command-line interface) builds an easy interface and functional usage for operating various services on AWS. But we are eager to make entire management much simpler and flexible to check the status, even put it aside until the job is finished. 

Of course, you have other choices like SDK for different languages, like [Python SDK](https://docs.aws.amazon.com/pythonsdk/?id=docs_gateway) or [Java SDK](https://docs.aws.amazon.com/sdk-for-java/?id=docs_gateway) if you like. In my experiences, every SDK sometimes use a different version of libraries or packages, you should be aware of the supporting version on each service you choose.

### Idea and Practice
We can choose the easy-checking cli command for pipeline, such as creating-cluster, add-steps, and terminate-clusters.
The cli command from document is shown below.
```shell
aws emr create-cluster \
--name "My First EMR Cluster" \
--release-label emr-5.31.0 \
--applications Name=Spark \
--ec2-attributes KeyName=myEMRKeyPairName \
--instance-type m5.xlarge \
--instance-count 3 \
--use-default-roles
```

Set the command as a variable to load in the next step, add 2 additional arguments `--query '[ClusterId]'`, `--output text` to change the output format and query the information we need, reference the variable `cluster_id` in the Shell scripts as below.
```shell
cluster_id="$(aws emr create-cluster --name SparkifyEMR \
--release-label emr-5.31.0 \
--log-uri "s3://$clustername/logs/" \
--applications Name=Spark \
--ec2-attributes KeyName="$keyname" \
--instance-type m5.xlarge \
--instance-count 3 \
--use-default-roles \
--query '[ClusterId]' \
--output text)"
echo -e "\nClusterId: $cluster_id"
```

Set the second variable `step_id` and query it.
```shell
step_id="$(aws emr add-steps \
--cluster-id "$cluster_id" \
--steps "Type=Spark,Name=SparkifyETL,ActionOnFailure=CONTINUE,\
Args=[s3://$clustername/run_etl.py,--data_source,$clustername/data/,--output_uri,$clustername/output/]" \
--query 'StepIds[0]' \
--output text)"
echo -e "\nStepId: $step_id"
```

Set a while loop to check Spark jobs is finished or not.
```shell
describe_step="$(aws emr describe-step --cluster-id "$cluster_id" \
  --step-id "$step_id" --query 'Step.Status.State' --output text)"

while true; do
  if [[ $describe_step = 'PENDING' ]]; then
    echo "Cluster creating... Autocheck state 2 mins later"
    sleep 120
  elif [[ $describe_step = 'RUNNING' ]]; then
    echo "Job running... Autocheck state 30 seconds later"
    sleep 30
  elif [[ $describe_step = 'FAILED' ]]; then
    echo "Job failed"
    break
  else
    echo "Job completed"
    break
  fi
done
```

After finishing all jobs, terminate the cluster of the jobs complete without error.
```shell
terminate_cluster="$(aws emr terminate-clusters \
--cluster-ids "$cluster_id" \
--query 'Cluster.Status.State' \
--output text)"

while true; do
  if [[ $terminate_cluster != 'TERMINATED' ]]; then
    echo "Cluster terminating..."
    sleep 15
  else
    echo "Cluster terminated"
  fi
done
```

### Conclusion
Anyway, this script brings you a convenient approach to run one-time Spark jobs in AWS EMR Cluster. You can also check the all lines in the script [here](https://gist.github.com/samuelTyh/04fb77ae0b81154d2a5b61b17d2635f8) for reference.

if you'd like to try it yourself, follow the tutorial from [AWS official documents](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html) to build and manage your EMR clusters and jobs.
