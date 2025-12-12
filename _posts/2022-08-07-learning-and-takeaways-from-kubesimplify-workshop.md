---
layout: post
title: Learning and Takeaways from Kubesimplify Workshop
description: https://kubesimplify.github.io/live-workshops/
image: 
category: [Learning Journey]
tags: devops linux docker k8s
date: 2022-08-07 00:03 +0200
---
This post will keep the learning process and takeaways from the [Kubesimplify workshops](https://kubesimplify.github.io/live-workshops/) held by [Saiyam Pathak](https://github.com/saiyam1814). The motivation for me to catch up on this DevOps topic is to systematically learning by practising for DevOps mindset, in order to achieve double blade stacks, facilitating Data Engineering tasks and projects via DevOps approaches.

# Workshops List
1. [Linux & Docker Fundamentals](#linux--docker-fundamentals) (Ongoing)
2. Kubernetes 101
3. GitOps With ArgoCD
4. Kubernetes Security 101
5. Kubernetes Troubleshooting


## Linux & Docker Fundamentals 
Instructor: [Chad M. Crowell](https://github.com/chadmcrowell)

- [Lecture&Workshop recording](https://youtu.be/EUu1E_YKGTw)
- [Learning resource](https://github.com/chadmcrowell/linux-docker)

### Linux fundamental
I've been familiar with most of the commands introduced in this session. I still learnt something new because my learning before wasn't systematical enough. The takeaways for me in this session are listed below:

1. Knowing the naming and how Linux filesystem works in a clear picture.
![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/linux_filesystem.png)
2. Pressing `control + r` in the prompt is able to search historical executions. Under z-shell, there's a way fancier drop-down list for a more intuitive search.
3. Using Command `man` to look up the manual of each command. `-h` isn't supported for every command, learnt how to use `man` is a great finding for me.
4. Linux commands are case-sensitive.
5. Simple command without switch on editor
```shell
echo 'var="something"' > file       # To overwrite the file
echo "var="something"' >> file      # To append to the file
```
6. Create an intermediate directory with `-p` flag while using `mkdir`, for example
```shell
mkdir test/sub-test/error           # The prompt would pop out error
mkdir -p test/sub-test/correct      # Successful execution
```
7. chmod commands usage [https://chmodcommand.com/chmod-600/](https://chmodcommand.com/chmod-600/)




**To be continued...**
