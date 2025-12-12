---
layout: post
title: Setting both Celery and Flask inside the docker-compose
description: 
image: 
category: [Learning Journey]
tags: celery flask docker
date: 2020-02-04 00:00 +0000
---
## Setting both Celery and Flask inside the docker-compose

Due to the issue I need to resolve is that put the heavy task to background, then run it periodically or asynchronously. And the most important thing is that have to support Python.
I found Celery, the excellent tool to implement distributed task queue by Python, can easily deal with request using asynchronous designing.
It seems like a perfect solution for my issue, however, there has still a new problem arised, what is I need to focus on processing the task ran by Flask and packed in docker and running by docker compose, it is difficult to find all of these resources at the same time on the Internet.
So let's started!

---
#### My article is referenced from the following resources:
1. [Asynchronous Tasks with Flask and Redis Queue](https://testdriven.io/blog/asynchronous-tasks-with-flask-and-redis-queue/)
2. [The web-service for extracting dominant color from images.](https://github.com/ViktorSalimonov/pylette?source=post_page---------------------------)
3. [Using Celery with Flask](https://blog.miguelgrinberg.com/post/using-celery-with-flask)
4. [Task schduling with Celery (Mandarin resource)](https://forster.site/2016-11-17-celery.html)

---
#### App Structure Displaying
```
Flask_Celery_example
|
├── api
|   ├── __init__.py
|   ├── app.py               
|   ├── celery_app.py
|   ├── config.py
|   ├── Dockerfile
|   └── requirements.txt
|
├── docker-compose.yml
└── README.md
```
#### Set up your Redis and Celery worker in docker-compose, the breaking point is to set up the celery worker's name well.
```yaml
version: "3"

services:
  web:
    container_name: web
    build: ./api
    ports:
      - "5000:5000"
    links:
      - redis
    depends_on:
      - redis
    environment:
      - FLASK_ENV=development
    volumes:
      - ./api:/app

  redis:
    container_name: redis
    image: redis:5.0.5
    hostname: redis

  worker:
    build:
      context: ./api
    hostname: worker
    entrypoint: celery
    command: -A celery_app.celery worker --loglevel=info
    volumes:
      - ./api:/app
    links:
      - redis
    depends_on:
      - redis

```

#### What the Dockerfile in this example looks like.
```docker
FROM python:3.6.5

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python", "./app.py" ]
```

#### Initialize Celery worker and create asynchronous or scheduled task in celery_app.py
```python
import config
from celery import Celery


# Initialize Celery
celery = Celery(
    'worker', 
    broker=config.CeleryConfig.CELERY_BROKER_URL,
    backend=config.CeleryConfig.CELERY_RESULT_BACKEND
)


@celery.task()
def func1(arg):
    ...
    return ...

```

#### Run task by Flask and check its status in app.py
```python
import config
from celery_app import func1
from flask import Flask, request, jsonify, url_for, redirect


# Your API definition
app = Flask(__name__)
app.config.from_object('config.FlaskConfig')

@app.route('/', methods=['POST'])
def longtask():
    task = func1.delay(arg)
    return redirect(url_for('taskstatus', task_id=task.id))
    
@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = func1.AsyncResult(task_id)
    if task.state == 'PENDING':
        time.sleep(config.SERVER_SLEEP)
        response = {
            'queue_state': task.state,
            'status': 'Process is ongoing...',
            'status_update': url_for('taskstatus', task_id=task.id)
        }
    else:
        response = {
            'queue_state': task.state,
            'result': task.wait()
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run()

```

#### After finishing the all setting, we can run the script on terminal below to create container and run it.
```shell
$ cd Flask_Celery_example
$ docker-compose build
$ docker-compose run
```

#### Conclusion
The bottleneck in this case is that I could run the Flask, Redis, Celery separately in docker-compose at first, but if I want to run it with only one script then it failed. I had tried and error lots of times to finding the breaking point, the correct script to run Celery in docker-compose. Everything was clear after throughing this bottleneck.
Thanks for reading my first article! Leave your message below my Facebook post if there's still any further questions. See you then.
