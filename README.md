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
The Riedel dataset (Riedel et al., 2010) used in this paper is publicly available. We provide the processed version of this dataset in addition to our manually labeled test set (data.zip). You can download it from this [Google Drive link](https://drive.google.com/drive/folders/1u2HVCYoJcV5SiFcmrP0yEIn5E5tJ6Mbg?usp=sharing) or this [Baidu Cloud link](https://pan.baidu.com/s/1anEw7xjmFZo6gaRWP0gpVw). With regard to the TACRED dataset, please refer to (Zhang et al., 2017).

### Files in data.zip
+ entity2id.txt: all entities and corresponding ids, one per line.

+ relation2id.txt: all relations and corresponding ids, one per line.

+ word2id.txt: words in the vocabulary and corresponding ids, one per line.

+ vec.txt: the pre-train word embedding file provided by (Lin et al., 2016).

+ word_embed.npy: the word embeddings for each word in the vocabulary.

+ train.txt: training set, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).

+ test.txt: the original test set, same format as train.txt.

+ manualTest.txt: the manually labeled test set.

## Code
The models are implemented using [Pytorch](https://pytorch.org/). 

### Files
+ data_loader.py: data processing.

+ model.py: implemented models, including sentence encoder, sentence selector, nEM framework.

+ main.py: model training and evaluation. Parameters setting.

+ run_RE.sh: the shell commands for baseline models' training and testing.

+ run_EM.sh: the shell commands for nEM model's training and testing.

### Sentence Encoders
+ CNN encoder.
+ GRU encoder.
+ PCNN encoder.

### Sentence Selectors
+ MEAN selector (Lin et al., 2016).
+ MAX selector (Jiang et al., 2016).
+ ATT selector (Lin et al., 2016). Note that our implemented attention-selector (refer to as multi-relational attention selector) is different from that one in (Lin et al., 2016). 

### The nEM Framework
+ E-step: computing a distribution on groud-truth labels.
+ M-step: updating model parameters through maxmizing the lower bound using gradient descent method.


## References

[Riedel et al., 2010] Sebastian Riedel, Limin Yao, Andrew McCallum. Modeling Relations and Their Mentions without Labeled Text. In Proceedings of ECMLPKDD.

[Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL.

[Jiang et al., 2016] Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang. Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks. In Proceedings of COLING.

[Zhang et al., 2017] Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, Christopher D. Manning. Position-aware Attention and Supervised Data Improve Slot Filling. In Proceedings of EMNLP.
