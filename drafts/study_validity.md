"Internal and external validity in observational studies: is my data set a fair comparison for an observational study"

[Blurb]

> often the only data available are observational and we must rely on the belief in subjective randomization, that is, there is no important variable that differs in the [treatment] trials and [control] trials.
> ...
> [T]he investigator should think hard about variables besides the treatment that may causally affect [the outcome] and plan in advance how to control for the important ones.

~http://www.fsb.muohio.edu/lij14/420_paper_Rubin74.pdf

# Intro: Intuition of a good observational analysis is a "fair comparison"

Checking for confounder overlap in observational studies

The intuitive reason we do experiments: it balances confounders

Sometimes we can't do an experiment, but was some experiment-like data, can we use that? we use regression or matching to control for confounders. but doing so is only value when the confounders overlap

this will cover methods of verifying whether important variables are balanced across the treatment and control sets; randomized experiments should pass this perfectly, but observational analyis may not be  as perfect

Mention the aircon example. intuitively, we want the aircon houses to be the same on everything except one variable

Fair comparison requires overlap

# Restricting to overlap areas and visual comparisons of C/T groups

types of variables:
* Treatment - assume one, binary
* Outcome - assume one, real
* Controls - maybe be real or categorical


Our example: Does adding aircon to a house change its price? Need to be careful of looking at same quality

https://replit.com/@srs_moonlight/overlap#main.py

# Impact ofd restriction: what samples were discarded?

# Final analysis (compare with unrestricted analysis

# Matching and "higher order fairness"

Validity checklist

Internal validity
[] Are all the relevant confounders included in the dataset? Are there any others you can come up with? Consult with the relevant domain experts. Build a DAG.
[] Balance checks
[] Do you need to restrict

External validity
[] Sample vs population comparisons
[] Changing external conditions

An elaboration - matching as preproc - https://gking.harvard.edu/files/matchp.pdf - Do CEM by coarsening, hashing, and then grouping.



