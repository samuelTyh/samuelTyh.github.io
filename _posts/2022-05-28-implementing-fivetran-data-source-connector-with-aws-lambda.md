---
layout: post
title: Implementing Fivetran Data Source Connector with AWS Lambda
description: 
image: 
category: [Data Engineering]
tags: aws lambda fivetran etl
date: 2022-05-28 22:47 +0200
---
## Basic background of AWS Lambda

[Official developer guide from AWS](https://aws.amazon.com/getting-started/hands-on/run-serverless-code/)

AWS Lambda is a serverless, event-driven compute service that lets you run code for virtually any type of application or backend service without provisioning or managing servers. In GCP and Azure, we can implement the same idea via Cloud Functions and Azure Funtions.


## Why is it necessary to understand Lambda to build Fivetran connectors?

Fivetran is a reliable data pipeline platform for business users to connect their data source in a convenient way. It provides tons of connectors and integrations for user to choose, like marketing tools, modern databases, etc. Although Fivetran supports amounts of external APIs and data sources integration in default, some of the external APIs we needin reality which doesn't be supported by Fivetran directly. 

If you are using AWS, then Lambda stands out at this moment! It allows us to write custom integration functions in Python, Node.js, etc. to approach data integration which not directly supported by Fivetran. In the following use case, connector/ETL was built in Python, connecting an API and Snwoflake. But this article would only focus on the less mentioned parts of the official Fivetran documentation. The basic architecture is shown in the figure below.


[ ![](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/Fivetran_lambda.drawio.png)](https://s3.eu-central-1.amazonaws.com/samueltyh.github.io/posts/Fivetran_lambda.drawio.png)


## Prerequisite

Follow the instruction from Fivetran to setup the configuration, click [here](https://fivetran.com/docs/functions/aws-lambda)

Lambda's sample function from Fivetran's document

```python
import json
def lambda_handler(request, context):
    # Fetch records using api calls
    (insertTransactions, deleteTransactions, newTransactionCursor) = api_response(request['state'], request['secrets'])    
    # Populate records in insert    
    insert = {}    
    insert['transactions'] = insertTransactions    
    delete = {}
    delete['transactions'] = deleteTransactions    
    state = {}
    state['transactionsCursor'] = newTransactionCursor    
    transactionsSchema = {}
    transactionsSchema['primary_key'] = ['order_id', 'date']    
    schema = {}
    schema['transactions'] = transactionsSchema    
    response = {}    
    # Add updated state to response
    response['state'] =  state    
    # Add all the records to be inserted in response
    response['insert'] = insert    
    # Add all the records to be marked as deleted in response
    response['delete'] = delete    
    # Add schema defintion in response
    response['schema'] = schema    
    # Add hasMore flag
    response['hasMore'] = 'false'    
    return response	
def api_response(state, secrets):
    # your api call goes here
    insertTransactions = [
            {"date":'2017-12-31T05:12:05Z', "order_id":1001, "amount":'$1200', "discount":'$12'},
            {"date":'2017-12-31T06:12:04Z', "order_id":1001, "amount":'$1200', "discount":'$12'},
    ]    
    deleteTransactions = [
            {"date":'2017-12-31T05:12:05Z', "order_id":1000, "amount":'$1200', "discount":'$12'},
            {"date":'2017-12-31T06:12:04Z', "order_id":1000, "amount":'$1200', "discount":'$12'},
    ]    
    return (insertTransactions, deleteTransactions, '2018-01-01T00:00:00Z')
```


## Multi-pages response implementation

Fivetran will stop the request if it gets a response has an attribute `hasMore` which equals `'false'`

```python
response['hasMore'] = 'false'
```

Which means, more than 1 pages response should be able to switch the value by a pointer. `false` should be able to altered to make Fivetran know that the request hasn't finished, the pointer should be updated as well once all pages are done. I'm sharing my implementation below to meet this requirement.

```python
import datetime
import asyncio
from services import processor, cursor_formatter


def handler(event, context):
    """
    Lambda function handler to handle output format
    """
    loop = asyncio.get_event_loop()
    insertLoanApplication, insertApplicants, insertAdvisors, state = \
        loop.run_until_complete(api_response(event['state'], event['secrets']))

    if not state['moreData']:
        loop.close()

    insert = {
        'loan_applications': insertLoanApplication,
        'applicants': insertApplicants,
        'advisors': insertAdvisors
    }

    schema_loan_applications = {'primary_key': ['id']}
    schema_applicants = {'primary_key': ['loan_application_id', 'id']}
    schema_advisors = {'primary_key': ['id']}
    schema = {
        'loan_applications': schema_loan_applications,
        'applicants': schema_applicants,
        'advisors': schema_advisors
    }

    return {
        'state': state if state['moreData'] else {'cursor': cursor_formatter(datetime.datetime.now()),
                                                  'page': 0},
        'insert': insert,
        'schema': schema,
        'hasMore': 'false' if not state['moreData'] else 'true'
    }


async def api_response(state, secrets):
    """
    Main function to call indicated API
    :param state: Fivetran anchor for indexing usage, default None
    :param secrets: The secret you would like to use to call the API
    :return: API responses
    """
    try:
        cursor_value, page = state['cursor'], state["page"]
    except KeyError:
        cursor_value, page = cursor_formatter(datetime.datetime.now()), 0
    page += 1
    insertLoanApplication, insertApplicant, insertAdvisor, moreData = await processor(secrets, page, cursor_value)
    state = {'cursor': cursor_value, 'page': page, 'moreData': moreData}

    return insertLoanApplication, insertApplicant, insertAdvisor, state
```

First, we can see in `api_request` function, `state` is assigned by request format, empty dictionary object in default, we can assign any pointer we need for requesting API, see [here](https://fivetran.com/docs/functions/aws-lambda/sample-functions#samplefunctionrequest) to check the detail.

We retrieve the `cursor` and `page` at the beginning, cursor is for locating the timestamp of each response whether we've done already, and page is literally for locating which page we are at. Fivetran can tell if this is an initial request, or if it is a request that is still pending and should be continued.

```python
try:
    cursor_value, page = state['cursor'], state["page"]
except KeyError:
    cursor_value, page = cursor_formatter(datetime.datetime.now()), 0
```

After processing, we will get a function return value `moreData` for the Lambda handler to continue or stop the request.

```python
state = {'cursor': cursor_value, 'page': page, 'moreData': moreData}
```

In this example, we have the return value from the handler to present continuing or stopping. Else we have no more new response from API, returning a timestamp and reset the page value to 0 for the next round requesting.

```python
return {
    'state': state if state['moreData'] else {'cursor': cursor_formatter(datetime.datetime.now()), 'page': 0},
    'insert': insert,
    'schema': schema,
    'hasMore': 'false' if not state['moreData'] else 'true'
}
```

## Conclusion

Since the usage context is relatively small, Fivetran does not have document that describes how to request a multi-page response from API requesting, so we use several pointers to implement the requirements alone, which still quite fits actually.

If you've noticed, I've also designed asynchronous requests for this purpose to speed up request efficiency, but this also creates a burden on the API server, I'll share how to optimize requests in a later post.
