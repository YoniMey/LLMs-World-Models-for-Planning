# Advanced Topics in Natural Language Processing - Final Project

This work is based on the paper: [Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning](https://guansuns.github.io/pages/llm-dm).

And the code is based on the repository: [LLMs-World-Models-for-Planning](https://github.com/GuanSuns/LLMs-World-Models-for-Planning)


## Expirement 1: PDDL task decomposition
In the paper, a single query was made for each new action, here we propose to break down the task into smaller sub-tasks. The task decomposition is done by the following steps:
1. Generate the parameters for the action.
2. Generate the preconditions for the action.
3. Generate the effects for the action.
