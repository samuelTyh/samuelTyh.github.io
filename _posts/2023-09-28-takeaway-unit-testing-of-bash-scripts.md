---
layout: post
title: 'Takeaway: Unit Testing of Bash Scripts'
description: 
image: 
category: [Learning Journey]
tags: bash unittest
date: 2023-09-28 13:40 +0200
---
Bash/Shell is a potent scripting language that lets us communicate with our computer's operating system. In many of my everyday tasks, I rely on Bash to carry out Linux commands and create certain logical processes. While Bash's capabilities are impressive, it's important to note that since it deals with low-level operations within the operating system, it can sometimes escalate minor problems into major ones, without any chance to revert the execution.

To ensure the reliability of our scripts for daily business operations, it's crucial to incorporate unit testing into our Bash scripts. This practice helps us maintain stability and prevent unexpected issues from affecting our workflow.

Briefly write down my takeaway from Chai Feng's sharing: https://www.youtube.com/live/-1gB_5dV32U?si=4uICRRG8vDmCKWxE&t=649



There are 3 main barriers that prevent us from implementing Bash unit tests.
- Dependencies, having the exact execution environment can be quite troublesome.
- The script could have potential side effects and can be tricky to set up or tear down.
- Command execution is the aim of the script, therefore it's hard to foresee the result.
- Execution/validation is not fast enough.

Let's revisit the concept of script’s unit tests. What exactly are they? 

Is it validating whether the function called runs smoothly? Or to validate the execution logic of the indicated script? It doesn’t matter running commands manually or by a script if the functions resulting errors. So the concept to implement scripts’ unit testing should be to validate the logic, instead of testing if the function can be run.

So what should we consider next? The methodology in Bash is essentially as same as other languages unit testing.
- Don't bother executing the commands in the PATH because they are external dependencies of the script.
- Each test instance is independent.
- Different platforms, different environments, same results.
- The validation action we expect is to execute the expected command, and send the expected parameters.

There are 5 things that determine the execution of commands. In order, they are aliases, keywords, functions, built-in procedures, and external procedures. The most important thing to consider about is how to simulate the command?


The testing framework for bash scripts: https://github.com/bach-sh/bach

