---
layout: post
title: Connect PostgreSQL in docker container with Azure Data Studio
description: 
image: 
category: [Learning Journey]
tags: azure-data-studio
date: 2020-02-05 00:00 +0000
---
## Connect PostgreSQL in docker container with Azure Data Studio

Azure Data Studio is a cool product that can easily connect MySQL (if you already installed in your system) and show what in your database look like. What if we use PostgresSQL instead of MySQL, how should we start?

The 1st step, I decided to pull Postgres's docker image but not to install at root.

```bash
$ docker pull postgres
```

Second, run the container, remember to set the port for local connecting (the document didn't mention it)
```bash
$ docker run --name postgres-docker -e POSTGRES_PASSWROD=secret_password -p 5432:5432 -d postgres
```

then you will get container id, also can check it by `docker ps`

```bash
CONTAINER ID        IMAGE                 COMMAND                  CREATED             STATUS              PORTS                      NAMES
ba01d334f4db        postgres              "docker-entrypoint.s…"   37 minutes ago      Up 37 minutes       0.0.0.0:5432->5432/tcp     postgres-docker
```

---
After running the container, we could start the next step, set up the connection in Azure Data Studio.

### 1. Install PostgreSQL plug-in.
Choose the extension's icon and search Postgres then install it.

[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/WukfL9H.png){:class="img-responsive"} ](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/WukfL9H.png)

### 2. Click new connection, choose PostgreSQL as your database.

[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/ACogtlC.png){:class="img-responsive"} ](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/ACogtlC.png)

### 3. Enter the required information.

[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/QCwXWTX.png){:class="img-responsive"} ](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/QCwXWTX.png)

### 4. Remember to set port in advanced setting.
Enter the ip and port as same as you assigned to docker container.

[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/n6oFLe2.png){:class="img-responsive"} ](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/n6oFLe2.png)

### 5. Connect and start to work in the database.

[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/V0rhy7k.png){:class="img-responsive"} ](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/V0rhy7k.png)

---
So far, the most important part is that remember to assign port when you run the docker container and enter the correct connection details, then it can keep your life much easier.
