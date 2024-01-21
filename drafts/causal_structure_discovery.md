# What we do and why

## What is a dag and why have one

## How data tells us about the causal graph

Simple example of a DAG

link shalizi here

## An example: housing prices

# How to do it

## Vars

## Constraints

## Inference

# Then what do you do with the graph

# Outline

https://github.com/py-why/causal-learn/blob/main/causallearn/search/ConstraintBased/PC.py

https://github.com/py-why/causal-learn/blob/main/tests/TestBackgroundKnowledge.py

https://github.com/py-why/causal-learn/blob/main/causallearn/utils/PCUtils/BackgroundKnowledge.py

to read: https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python?tab=readme-ov-file

https://arxiv.org/abs/2307.16405

Big idea: (1) Learn graph from causal-learn, output as nx (2) Input to dowhy to get confounders, do estimates 
