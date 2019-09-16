# nEM
Code and data for EMNLP2019 Paper ["Uncover the Ground-Truth Relations in Distant Supervision: A Neural Expectation-Maximization Framework"](https://arxiv.org/pdf/1909.05448.pdf).

## Description

This work focuses on the noisy-label problem in distant supervision, while most of the previous works in this setting assume that the labels of a bag is clean and what makes the noisy nature of distant-supervised data is that the sentences in the bag may be noisy. Through a careful investigation of the distant-supervised data sets, we argue that both sentences and labels can be noisy in a bag. The following figure uncovers the cause of noise in distant supervision.

<p align="center">
  <img src="https://github.com/AlbertChen1991/nEM/blob/master/fig/noise.png">
</p>

In distant supervision, one attempts to connect labels of a bag (left side in the figure) with relations of the knowledge graphs (right side in the figure). The ground-truth labels (center in the figure) however can not be seen. The noise occurs due to the gaps between the bags and the knowledge graphs. For example, the bag in the above figure will be assigned three labels (r2, r3, r4). However, r4 is an obviously wrong label for this bag since all the sentences (s1, s2, s3) don't support r4. And r1 is a missing label for this bag because s1 support relation r1. These two cases are so-called noisy-label problem. The extensively studied noisy-sentence problem is also reflected in the figure. For example, s1 and s3 are noisy sentences for bag label r2, since only s2 supports relation r2.

We proposed a nEM framework to deal with the noisy-label problem. We manually labeled a subset of the test set of the Riedel dataset (NYT). The following figure shows the evaluation result on this clean test set. The baselines are PCNN+MEAN (Lin et al., 2016), PCNN+MAX (Jiang et al., 2016) and PCNN+ATT (Lin et al., 2016)

<p align="center">
  <img src="https://github.com/AlbertChen1991/nEM/blob/master/fig/PR_curve.png">
</p>

## Dataset
The Riedel dataset (Riedel et al., 2010) used in this paper is publicly available. We provide the processed version of this dataset. You can download it from this [Google Drive link](aaa.com) or [Baidu Cloud link](https://pan.baidu.com/s/1anEw7xjmFZo6gaRWP0gpVw)

## References
[Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL.

[Jiang et al., 2016] Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang. Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks. In Proceedings of COLING.

[Riedel et al., 2010] Sebastian Riedel, Limin Yao, Andrew McCallum. Modeling Relations and Their Mentions without Labeled Text. In Proceedings of ECMLPKDD.

