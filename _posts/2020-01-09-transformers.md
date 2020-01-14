---
layout: post
slug: bert-secrets
author: arogers
title:  "The Dark Secrets of BERT"
date:   2020-01-07 21:00:47
tags: transformers 
mathjax: false
toc: true
excerpt: "BERT and its Transformer-based cousins are still ahead on all NLP leaderboards. But how much do they actually understand about language?"
header:
    og_image: /assets/images/bert-header.png
---

> This blog post summarizes our EMNLP 2019 paper "Revealing the Dark Secrets of BERT" {% cite KovalevaRomanovEtAl_2019_Revealing_Dark_Secrets_of_BERT %}. Paper PDF: [https://www.aclweb.org/anthology/D19-1445.pdf](https://www.aclweb.org/anthology/D19-1445.pdf) 

2019 could be called the year of the Transformer in NLP: this architecture [dominated the leaderboards](https://hackingsemantics.xyz/2019/leaderboards/) and inspired many analysis studies. The most popular Transformer is, undoubtedly, BERT {% cite DevlinChangEtAl_2019_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding %}. In addition to its numerous applications, multiple studies probed this model for various kinds of linguistic knowledge, typically to conclude that such knowledge is indeed present, to at least some extent {% cite Goldberg_2019_Assessing_BERTs_Syntactic_Abilities HewittManning_2019_Structural_Probe_for_Finding_Syntax_in_Word_Representations Ettinger_2019_What_BERT_is_not_Lessons_from_new_suite_of_psycholinguistic_diagnostics_for_language_models %}. 

This work focuses on the complementary question: what happens in the *fine-tuned* BERT? In particular, how much of the linguistically interpretable self-attention patterns that are presumed to be its strength are actually used to solve downstream tasks?

To answer this question, we experiment with BERT fine-tuned on the following GLUE {% cite WangSinghEtAl_2018_GLUE_A_Multi-Task_Benchmark_and_Analysis_Platform_for_Natural_Language_Understanding %} tasks and datasets: 

* paraphrase detection (MRPC and QQP);
* textual similarity (STS-B);
* sentiment analysis (SST-2);
* textual entailment (RTE);
* natural language inference (QNLI, MNLI).

## A brief intro to BERT

BERT stands for Bidirectional Encoder Representations from Transformers. This model is basically a multi-layer bidirectional Transformer encoder {% cite DevlinChangEtAl_2019_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding %}, and there are multiple excellent guides about how it works generally, including [the Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). What we focus on is one specific component of Transformer architecture known as self-attention. In a nutshell, it is a way to weigh the components of the input and output sequences so as to model relations between them, even long-distance dependencies. 

As a brief example, let's say we need to create a representation of the sentence "Tom is a black cat". BERT may choose to pay more attention to "Tom" while encoding the word "cat", and less attention to the words "is", "a", "black". This could be represented as a vector of weights (for each word in the sentence). Such vectors are computed when the model encodes each word in the sequence, yielding a square matrix which we refer to as the self-attention map.

Now, a priori it is not clear that the relation between "Tom" and "cat" is always the best one. To answer questions about the color of the cat, a model would do better to focus on "black" rather than "Tom". Luckily, it doesn't have to choose. The power of BERT (and other Transformers) is largely attributed to the fact that there are multiple heads in multiple layers that all learn to construct independent self-attention maps. Theoretically, this could give the model the capacity to "attend to information from different representation subspaces at different positions" {% cite VaswaniShazeerEtAl_2017_Attention_is_all_you_need %}. In other words, the model would be able to choose between several alternative representations for the task at hand.

Most of the computation of self-attention weights happens in BERT during pre-training: the model is (pre-)trained on two tasks (masked language model and next sentence prediction), and subsequently fine-tuned for individual downstream tasks such as sentiment analysis. The basic idea for this separation of the training process into semi-supervised pre-training and supervised fine-tuning phases is that of transfer learning: the task datasets are typically too small to learn enough about language in general, but large text corpora can be used for this via the language modeling objective (and other similar ones). We could thus get task-independent, but informative representations of sentences and texts, which then could be "adapted" for the downstream tasks. 

Let us note here that the exact way the "adaptation" is supposed to work is not described in detail in either BERT paper or the GPT technical report (which highlighted the pre-training/fine-tuning approach). However, if attention itself is meant to provide a way to "link" parts of the input sequence so as to increase its informativeness, and the multi-head, multi-layer architecture is needed to provide multiple alternative attention maps, presumably the fine-tuning process would teach the model to rely on the maps that are more useful for the task at hand. For instance, one could expect that relations between nouns and adjectives are more important for sentiment analysis task than relations between nouns and prepositions, and so fine-tuning would ideally teach the model to rely more on the more useful self-attention maps.

## What types of self-attention patterns are learned, and how many of each type?

So what are the patterns of the self-attention in BERT? We found five, as shown below:

<figure>
	<img src="{{'/assets/images/bert-attn-types.png' | relative_url }}"> 
	<figcaption>Fig. 1. Types of self-attention patterns in BERT. Both axes on every image represent BERT tokens of an input example, and colors denote absolute attention weights (darker colors stand for greater weights).
	</figcaption>
</figure>

* The vertical pattern indicates attention to a single token, which usually is either the [SEP] token (special token representing the end of a sentence), or [CLS] (special BERT token that is used as full sequence representation fed to the classifiers).
* The diagonal pattern indicates the attention to previous/next words;
* The block pattern indicates more-or-less uniform attention to all tokens in a sequence;
* The heterogeneous pattern is the only pattern that theoretically *could* correspond to anything like meaningful relations between parts of the input sequence (although not necessarily so).

And here are the ratios of these five types of attention in BERT fine-tuned on seven GLUE tasks (with each column representing 100% of all heads in all layers):  

<figure>
	<img src="{{'/assets/images/bert-attn-ratios.png' | relative_url }}"> 	
	<figcaption>Fig. 2. Ratios of self-attention map types for BERT fine-tuned on the selected GLUE tasks.
	</figcaption>
</figure>

While the exact ratios vary by the task, in most cases the potentially meaningful patterns constitute less than half of all BERT self-attention weights. At least a third of BERT heads attends simply to [SEP] and [CLS] tokens - a strategy that cannot contribute much of meaningful information to the next layer's representations. It also shows that the model is severely overparametrized, which explains the recent successful attempts of its distillation {% cite SanhDebutEtAl_2019_DistilBERT_distilled_version_of_BERT_smaller_faster_cheaper_and_lighter JiaoYinEtAl_2019_TinyBERT_Distilling_BERT_for_Natural_Language_Understanding%}.

Note that we experimented with BERT-base, the smaller model with 12 heads in 16 layers. If it is already so overparametrized, this has implications for BERT-large and all the later models, some of which are 30 times larger {% cite WuSchusterEtAl_2016_Googles_neural_machine_translation_system_bridging_gap_between_human_and_machine_translation %}.

Such reliance on [SEP] and [CLS] tokens could also suggest that either they somehow "absorb" the informative representations obtained in the earlier layers, and subsequent self-attention maps are simply not needed much, or that BERT overall does not rely on self-attention maps to the degree to which one would expect for this key component of this architecture. 

## What happens in fine-tuning?

Our next question was what actually changes during the fine-tuning of BERT. The heatmap below shows the cosine similarities between flattened self-attention map matrices in each head and each layer, before and after fine-tuning. Darker colors indicate more differences in the representation. For all GLUE tasks the fine-tuning was done for 3 epochs.

<figure>
	<img src="{{'/assets/images/bert-finetune-diff.png' | relative_url }}"> 		
	<figcaption>Fig. 3. Cosine similarity between flattened self-attention maps, per head in pre-trained and fine-tuned BERT. Darker colors indicate greater differences.
	</figcaption>
</figure>

 We see that most attention weights do not change all that much, and for most tasks, the last two layers show the most change. These changes do not appear to favor any specific types of meaningful attention patterns. Instead, we find that the model basically learns to rely more on the vertical attention pattern. In the SST example below the thicker vertical attention patterns in the last layers are due to the joint attention to the final [SEP] and the punctuation tokens preceding it, which we observed to be another frequent target for the vertical attention pattern.
 
 <figure>
	<img src="{{'/assets/images/bert-sst-heads.png' | relative_url }}"> 			
	<figcaption>Fig. 4. Self-attention maps for an individual example, with BERT fine-tuned on SST.
	</figcaption>
</figure>
 
 This has two possible explanations: 
 
 * the vertical pattern is somehow sufficient, i.e. the [SEP] token representations somehow absorbed the meaningful attention patterns from previous layers. We did find that the earliest layers attend to [CLS] more, and then [SEP] starts dominating for most tasks (see Fig. 6 in the paper);
 * the tasks at hand do not actually require the fine-grained meaningful attention patterns that are supposed to be the main feature of the Transformers.

## How much difference does fine-tuning make?

Given the vast discrepancies in the size of the datasets used in pre-training and fine-tuning, and the very different training objectives, it is interesting to investigate how much of a difference fine-tuning actually makes. To the best of our knowledge, this question has not been addressed before.

We conduct three experiments on each of the selected GLUE datasets: 

* BERT performance with weights frozen from pre-training and fed to the task-specific classifiers;
* BERT performance with a model randomly initialized from normal distribution, and fine-tuned on task datasets for 3 epoches;
* BERT performance with the official pretrained BERT-base model, fine-tuned on task datasets for 3 epochs.  

The results of this experiment were as follows:

<div class="table-wrapper" markdown="block">

| Dataset| Pretrained|Random+finetuned | Pretrained+finetuned | Metric | Dataset size |
|--------|-----------|-----------|-----------|--------|------|
|MRPC    | 31.6      | 68.3      | 82.3      | Acc    | 5.8K |
|STS-B   | 33.1      | 2.9       | 82.7      | Acc    | 8.6K |
|SST-2   | 49.1      | 80.5      | 92        | Acc    | 70K  |
|QQP     | 60.9      | 63.2      | 78.6      | Acc    | 400K |
|RTE     | 52.7      | 52.7      | 64.6      | Acc    | 2.7K |
|QNLI    | 52.8      | 49.5      | 84.4      | Acc    | 130K |
|MNLI-m  | 31.7      |  61.0     | 78.6      | Acc    | 440K |

</div>

While it is clear that pretraining + fine-tuning setup yields the highest results, the **random + fine-tuned BERT is doing disturbingly well on all tasks except textual similarity**. Indeed, for sentiment analysis it appears that one could get 80% accuracy with randomly initialized and fine-tuned BERT, without any pre-training. Given the scale of the large pre-trained Transformers, this raises serious questions about whether the expensive pre-training yields enough bang for the buck. It also raises serious questions about the NLP datasets that apparently can be solved without much task-independent linguistic knowledge that the pre-training + fine-tuning setup was supposed to deliver. 


## Are there any linguistically interpretable self-attention heads?

Several studies at this point tried to locate self-attention heads that encode specific types of information, but most of them focused on syntax. We conducted an experiment focusing on frame semantic elements: we extracted from FrameNet 1.7 {% cite BakerFillmoreEtAl_1998_Berkeley_Framenet_project %} 473 sentences which were at most 12 tokens in length (to reduce the number of sentences evoking multiple frames), and which had a core frame element at a distance of at least 2 tokens from the target word (irrespective of its syntactic function). In the example below, it is the relation between the Experiencer and the participle "agitated" that evokes the Emotion_directed frame. Arguably, such relations are indeed core to understanding the situation described by a given sentence, and any mechanism claiming to provide linguistically informative self-attention maps should reflect them (possibly among many other relations). 

We obtained representations of these sentences by pre-trained BERT, calculating the maximum weights between token pairs corresponding to the annotated frame semantic relations. Fig. 5 represents the averages of these scores for all examples in our FrameNet dataset. We found two heads (head 2 in layer 1, head 6 in layer 7) that attended to these frame semantic relations more than the other heads.

<figure>
	<img src="{{'/assets/images/bert-frames.png' | relative_url }}"> 				
	<figcaption>Fig. 5. The heads of pre-trained BERT that appear to encode the information correlated to semantic links in the input text.
	</figcaption>
</figure>

## But... what information actually gets used at inference time?

We believe it would be too rash to conclude from probes of pre-trained BERT weights that certain information is actually encoded. Given the size of the model, it might be possible to find similar proof of encoding for any other relation (and the fact that Jawahar et al. found no significant difference between different decomposition schemes points in that direction {% cite JawaharSagotEtAl_2019_What_does_BERT_learn_about_structure_of_language %}). The real question is whether the model learns to actually rely on that information at inference time. 
 
To see whether the two heads we identified as useful for encoding frame semantic relations actually get used by fine-tuned BERT, we performed an ablation study, disabling one head at a time (i.e. replacing the learned attention weights with uniform attention). Fig. 6 shows a heatmap for all GLUE tasks in our sample, with each cell indicating the overall performance when a given head was switched off. It is clear that while the overall pattern varies between tasks, on average we are better off removing a random head - including those that we identified as encoding meaningful information that should be relevant for most tasks. Many of the heads can also be switched off without any effect on performance, again pointing at the fact that even the base BERT is severely overparametrized.

<figure>
	<img src="{{'/assets/images/bert-ablate-heads.png' | relative_url }}"> 			
	<figcaption>Fig. 6. Performance of the model while disabling one head at a time. The orange line indicates the baseline performance with no disabled heads. Darker colors correspond to greater performance scores.
	</figcaption>
</figure>

Similar conclusions were reached independently for machine translation task, with zeroing attention weights rather than replacing them with uniform attention {% cite MichelLevyEtAl_2019_Are_Sixteen_Heads_Really_Better_than_One %}. We further show that this observations extends not to just heads, but whole layers: depending on the task, a whole layer may be detrimental to the model performance! 

<figure>
	<img src="{{'/assets/images/bert-ablate-layers.png' | relative_url }}"> 			
	<figcaption>Fig. 7. Performance of the model while disabling one layer at a time.
	</figcaption>
</figure>

## So how does this thing work?

To sum up, this work showed that even base BERT is severely overparametrized, which explains why model distillation turned out to be so productive. 

Our key contribution is that while most studies of BERT focused on probing the pre-trained model, we raised the question of what happens in fine-tuning, and just how meaningful the representations obtained with the self-attention mechanism are. We were unable to find any evidence of linguistically meaningful self-attention maps being crucial for the performance of fine-tuned BERT.

These results could be interpreted in the following ways:

 a) **BERT is overparametrized**: Since we switch off only one head at a time, it may be possible that some heads are functional duplicates, and removing one head would not harm the model because the same information is available elsewhere. That would again point at overparametrization and importance of model distillation: with a large model, it is not feasible to test this hypothesis by switching off all possible combinations of heads. 
 
 b) **BERT's success is due to ~~black magic~~ something other than self-attention maps**: The information we as humans deem important for solving a verbal reasoning task may genuinely be not needed by the model, as it performs some deeper reasoning we are not able to comprehend or interpret (perhaps relying on some other component than the interpretable self-attention maps).
 
 c) **BERT does not need to be all that smart for these tasks**: The model does not actually solve the verbal reasoning task, but learns to rely on various shortcuts, biases and artifacts in the datasets to arrive at the correct prediction, and therefore does not need the attention maps to be particularly informative. 

It is possible that all three of the above factors are playing a role. However, given our findings about how well a randomly initialized BERT does on most GLUE tasks, and the recent discoveries of problems with many current datasets{% cite GururanganSwayamdiptaEtAl_2018_Annotation_Artifacts_in_Natural_Language_Inference_Data McCoyPavlickEtAl_2019_Right_for_Wrong_Reasons_Diagnosing_Syntactic_Heuristics_in_Natural_Language_Inference %}, the easy dataset factor seems to be very likely.

{% include bib_footer.markdown %}
