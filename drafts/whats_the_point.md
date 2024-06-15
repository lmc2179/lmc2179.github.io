---
layout: post
title: "What's the point of data scientists?"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*Data science is a field with wide scope and tons of fancy tools. But don't be distracted - data science is fundamentally about using data to improve decision making. This is the common thread underlying all useful data analysis activities, and keeping that in mind will make you a better data scientist.*

>**Statisticians produce solutions to problems.** ... One cannot answer the question “what is important?” without first considering the question “important for what?”

--Giovanni Parmigiani, _Decision Theory_, On one possible way of describing the work of data analysis

> There is, then, no single, all-purpose, correct answer to a question implying measurement unless we specify the relevant local concerns that give rise to the question. 

--James C. Scott, _Seeing Like a State_, On measuring what matters to the circumstances at hand



# The most overhyped job of the 21st century

https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century

1. You give the data scientist access to your database.
2. The data scientist logs in, and performs a series of mystic incantations. They ✨deep learn the big data✨. They Hadoop the random forests. They write a bunch of unreadable equations on your office whiteboard, nod knowingly as they look out at the horizon, and smile as the immutable truths of the universe wash over them.
3. Revenue doubles overnight. Costs go to zero. Your company has an IPO for a bazillion dollars, and you end up on the cover of _Extremely Rich Tech Executive Magazine_, right next to the story about personal jets powered by bitcoin.


# Step 1: Measure

James C Scott - Use what is to hand, be as exact as is needed. Make the system legible
> Rainfall may be said to be abundant or inadequate if the context of the query implies an interest in a particular crop. (...) Local measures were also relational or "commensurable.

Connect to your north star

Collect data. Define outcomes, levers, background variables. Create monitoring systems. Propose DAGs. Set priors. Ensure data quality.

# Step 2: Learn

Tukey - Make a model of the world you can commit to. Draw conclusions

Key levers? Do experiments

Come to scientific conclusions. Estimate. Quantify uncertainty. Qualitatively describe. Rule out/falsify DAGs. Fit models. Simulate parameter space. Locate insights.

# Step 3: Decide

Parmigiani/Savage

Pick the actions : facts -> outcomes

Consider cost + uncertainty. Simulate. Pick a course of action with the best info you have. Simulate outcome space.

# APPENDIX: Original draft

# So what is it that data scientists do?

Googling give us some almost-right answers. Various articles, blog posts, and other sources, tell us that data scientists are important because they...
- ...produce insights about the data
- ...use tools that make giant data sets manageable
- ...find and explain patterns
- ...build a theory of the business
- ...build and deploy models
- ...help the business test its ideas with experimentation

All of these are quite decent answers. Data scientists in different setting definitely do all these things - as a working data scientists, each of these is part of my professional life, and I agree they are important. But in the pre Data Science era, many businesses got along just fine without these things.  

The point of data scientists - the reason we do all the things in the list above - is to use data to _make better decisions_.

# The mantra of the pragmatic data scientist: "What's the decision?"

We sort important from useless information by thinking about whether the information is relevant to the decision at hand

If you can't think of a decision that your analysis or data would benefit from, then don't do the analysis

Good DS think about decisions, Great DS think about decision processes - the entire feedback loop encompassing data collection, analysis, recommendation, and selection

# Good data scientists: How do I help make better decisions?

> The task of decision involves three steps: (1) The listing of all the alternative strategies; (2) the determination of all the consequences that follow upon each of these strategies; (3) the comparative evaluation of these sets of consequences.
> (...) The behaving subject cannot, of course, know directly the consequences that will follow upon their behavior. (...) What they do is to form _expectations_ of future consequences, these expectations being based upon known empirical relationships, and upon information about the existing situation.

--Herbert Simon, _Administrative Behavior_, Ch. IV, describing an idealized decision-making process

when do I need higher fidelity data? if knowing it would help my decision



# Great data scientists: How do I design better repeatable decision processes?

industrial beekeeping

Decision process
Analysis --> Choice --> Evaluation --> Improvement

which accept feedback and scale with the organization - the goal is not to create the one true playbook, but to deal effectively with 80% and create flexible frameworks for the rest

# Full stack data ownership

[Eng] Prod -> F/D -> Agg/Metrics -> Model/Exp -> Decision [PM]

# Sidebar: What about other, similar job titles? Data analysts? Statisticians? ML engineers? Business analysts?

It's all the same

# An important consequence: Good organizational decision making means effective analysis _and_ communication

# An example of good data science: Scaling A/B testing

I want to finish by telling a story that's about a topic that's near to my heart - A/B testing. It is a case study in successful scaling instantiation and scaling of a decision process
