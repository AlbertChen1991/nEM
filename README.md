# nEM
Code for EMNLP2019 Paper "Uncover the Ground-Truth Relations in Distant Supervision: A Neural Expectation-Maximization Framework".

## Description

This work focuses on the noisy-label problem in distant supervision, while most of the previous works in this setting assume that the labels of a bag is clean and what makes the noisy nature of distant-supervised data is that the sentences in the bag may be noisy. Through a careful investigation of the distant-supervised data sets, we argue that both sentences and labels can be noisy in a bag. The following figure uncovers the cause of noise in distant supervision.

<p align="center">
  <img src="https://github.com/AlbertChen1991/nEM/blob/master/fig/noise.png">
</p>

In distant supervision, one attempts to connect labels of a bag (left side in the figure) with relations of the knowledge graphs (right side in the figure). The ground-truth labels (center in the figure) however can not be seen. The noise occurs due to the gaps between the bags and the knowledge graphs. For example, 
