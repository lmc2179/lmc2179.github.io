---
layout: post
title: "How to stop an experiment early with alpha spending"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: alpha_spending.png
---




https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/LanDeMets.pdf

def obf_alpha(t_prop): return 2 - 2*(norm.cdf(1.96/np.sqrt(t_prop)))
