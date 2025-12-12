---
layout: post
title: Useful gadget sharing - cron-job.org
description: 
image: 
category: [Sharing]
tags: heroku cron
date: 2021-04-30 00:00 +0000
---
# Useful gadget sharing - cron-job.org

Due to initiating to maintain my side projects which have done before, I started investigating any pain points that needed to be improved or could be divided into small projects.

## Story

I want to introduce a really cool, simple but useful gadget for you, cron-job.org.

The context was that I had a side project which deployed on Heroku. Heroku is a really convenient platform that provides a basic deployment environment for someone who wants to build their website or front-end app interface.

In short, Heroku free tier is quite sufficient for me. But your app is put to down after 30 mins of inactivity, and will need around 5–10 seconds to wake up the app again. If your app cannot tolerate that in any circumstance, you may want to try some approaches to avoid your app sleep again but still free. Pinging your app on a set interval might be an economy way to achieve it.

## Approach

1. Make sure you have a simple GET request to your app homepage. For example, if your app is hosted at `your-app.herokuapp.com`, make sure you have a GET method handling **“/”** route. Or, have a GET method at some other route.
2. Once that’s completed, we can either handle pinging our own app manually or automate the process. The Heroku free tier requires applications to sleep a minimum of seven hours every day. Unverified accounts get 550 of free dyno hours per month, which is equal to almost 7 hours of sleep time for a month of 31 days. To be on the safe side, we’ll go with seven hours of sleep.
3. The example GET function writing in Python/Flask

```python
@bp.route('/ping', methods=['GET'])
def ping():
    res = {
        "name": "CV parser",
        "requested_at": datetime.datetime.now(),
        "status": "ok"
    }
    return jsonify(res)
```

## Using [cronjob.org](http://cronjob.org) to keep your app alive

- Signup on [https://cron-job.org/en/](https://cron-job.org/en/)
- Verify your email once the signup is done.
- Log in and click on the Cronjobs tab. And click on “Create cronjob”.
- Fill in the details. Enter URL according to your GET method. For example, my URL will be `https://you.herokuapp.com/ping`

![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/schedule_example.mov)

Schedule the execution time from 8 to 23 every 30 minutes every day

- You can also check the execution detail in dashboard

[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/cronjob_dashboard.png){:class="img-responsive"} ](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/cronjob_dashboard.png)
