# nEM
Code for EMNLP2019 Paper "Uncover the Ground-Truth Relations in Distant Supervision: A Neural Expectation-Maximization Framework".

## Description

This work focuses on the noisy-label problem in distant supervision, while most of the previous works in this setting assume that the labels of a bag is clean and what makes the noisy nature of distant-supervised data is that the sentences in the bag may be noisy. Through a careful investigation of the distant-supervised data sets, we argue that both sentences and labels can be noisy in a bag. The following figure uncovers the cause of noise in distant supervision.

<p align="center">
  <img src="https://github.com/AlbertChen1991/nEM/blob/master/fig/noise.png">
</p>

In distant supervision, one attempts to connect labels of a bag (left side in the figure) with relations of the knowledge graphs (right side in the figure). The ground-truth labels (center in the figure) however can not be seen. The noise occurs due to the gaps between the bags and the knowledge graphs. For example, the bag in the above figure will be assigned three labels (r2, 23, r4). However, r4 is an obviously wrong label for this bag since all the sentences (s1, s2, s3) don't support r4. And r1 is a missing label for this bag because s1 support relation r1. These two cases are so-called noisy-label problem. The extensively studied noisy-sentence problem is also reflected in the figure. For example, s1 and s3 are noisy sentences for bag label r2, since only s2 supports relation r2.
