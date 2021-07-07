---
layout: post
slug: theory-1
author: vlialin
title:  "What Deep Learning Theories Can Tell Us Today"
date:   2020-08-31
tags: theory
mathjax: true
toc: true
excerpt: "Deep learning theory is a new and rapidly evolving field of statistical learning. In this blog post, we try to summarize a part of it."
og_image: /assets/images/gen_photo.png
---


# What Deep Learning Theories Can Tell Us Today

Deep learning has been with us for a while now and I expect that many of you who decided to read this post can
easily build and train a network for images or text classification.
For me it has been quite a journey in NLP for the last 4 years - we had LSTMs, then CNNs, then pre-trained
language models, and then BERT et al.
who overthrown most other methods in just a couple of months.
Even though some people, especially people who are not familiar with deep learning, can say that nobody understands
how and why these huge networks make their predictions, there is quite some success in this area.
There are many ways to visualize feature importance and to find model weaknesses, some of them are more
complicated than others, but in general the area of qualitative analysis of trained models is developing
pretty fast and the community asks (and answers) a lot of the right questions.

However, a full understanding of the methods is impossible without the math side of it.
Some classical analysis and intuition that suit linear models and SVMs well, fail miserably in deep learning.
And this does not say that the math is wrong or that deep learning does not work and all we can see around us is a very unlikely event.
It only means that some of the suggestions that the ML community of the past made are either too general and do not consider neural network specifics
or that some of them fail at particular scales.
So it means that we need new math, which is more specific to neural networks, and that considers the scale of the datasets and models that we work with.
And many great mathematicians work on it.

In this small series of posts, we will talk about the state of modern deep learning theories that try to explain deep learning
success and to get some new insights about it.
We will mostly discuss the results omitting the proofs and precise theorem formulations.
This series is written by practice people who are interested in what is happening in theory and 
we would not claim that it reviews most of the key theoretical works that shape our new vision of deep learning,
but we like to mention some of the results we personally find interesting and important.

We want to start with the question of why we can optimize neural networks using simple gradient descent methods
and not to be stuck in some bad local minimum.

## How bad is this local minimum?

Let’s look at a picture you probably saw in your university classes.
It is about gradient descent and convex optimization.

<figure>
   <img src="https://www.researchgate.net/profile/Md_Saiful_Islam14/publication/338621083/figure/fig4/AS:847811214069760@1579145353037/Gradient-Descent-Stuck-at-Local-Minima-18.ppm"> 
   <figcaption>Fig. 1. Gradient descent stuck at local minima (<a href="https://www.researchgate.net/figure/Gradient-Descent-Stuck-at-Local-Minima-18_fig4_338621083">image source</a>).
   </figcaption>
</figure>

Gradient descent is greedy.
It only improves your loss locally and it is commonly accepted that GS is guaranteed to converge to a global minimum
only in a convex setup.
On the other hand, the neural network loss landscape is highly complex and nonconvex.
So probably we should not use it, right?

Yet the practice tells us otherwise, it shows that local optimization is surprisingly
effective for neural networks.

<figure>
   <img src="{{'/assets/images/theory1_microsoft_loss_landscape.png' | relative_url }}"> 
   <figcaption>Fig. 2. Loss landscape of transformer networks (<a href="https://arxiv.org/abs/1908.05620">Hao et al., 2019</a>)
   </figcaption>
</figure>

The researchers who did neural networks before the era of deep learning, including
[Yann LeCun](](https://lexfridman.com/yann-lecun/), say that the fact that NNs can be
optimized using gradient descent fascinates them and people would not expect it to
work back in the days.

But it does.
And why is not the question of applied DL, but is the question of theory.
Take linear (or logistic) regression which is the simplest kind of neural network one
can find. Linear regression loss is just $$(y - Ax)^2$$ which is simple and convex and only
has one local minimum - its global minimum (just like the local minima on Fig. 1).

[Lu and Kawaguchi (2017)](https://arxiv.org/abs/1702.08580)
studied a bit more complex form of a neural network.
Linear neural network - one that does not have nonlinear activation functions -
$$A_N A_N-1 … A_1 x$$.
It is easy to see that such a network, just like linear regression, can only describe
linear functions.
Yet, the optimization problem is not trivial and the loss-landscape is not convex. 
heir result is quite fascinating. These networks have multiple local minima,
but they are all the same.
So even if you always go the steepest curve you find the lowest valley!

A similar result has been known for some time in wide (enough) shallow networks with
sigmoid nonlinearities
[(Yu and Chen, 1995)](https://arxiv.org/abs/1704.08045).
Going further,
[Nguyen and Hein (2017)](https://arxiv.org/abs/1704.08045)
generalized it to deep sigmoid networks
(with the layer size shrinking down after some point).
And very recently it was shown that a neural network with ReLU activations and a "wide enough"
hidden layer has all of its global minima connected [[Nguyen, 2019](https://arxiv.org/abs/1901.07417).
For a visual (and very simplified) example, look at the picture below.

<figure>
   <img src="{{'/assets/images/theory1_sublevel_sets.png' | relative_url }}"> 
   <figcaption>Fig. 3. A non-convex function with connected global minima (<a href="https://arxiv.org/abs/1901.07417">Nguyen, 2019</a>)
   </figcaption>
</figure>

Now let’s take a few steps back and think about this.
The progress in understanding loss landscape that the theory community made in the last
few years is tremendous.
However, an important part is that even if we have only global minima,
this does not mean that gradient descent can find even one of them in a reasonable time.
Imagine an extreme case where you have a huge (more than your learning rate) and a
completely flat plateau.
As soon as your optimizer leads you there, you are stuck forever.

## Where do you lead me (S)GD?

So we can expect from the loss landscape of NNs to be even more special.
One of the examples is that, not only linear networks do not have bad local minima,
but also you can avoid saddle points near initialization if you use residual connections
[(Hardt and Ma, 2016)](https://arxiv.org/abs/1611.04231).
And the reason is simple, a network with residual connections at forward pass is just a
specific kind of a feedforward network.
In the simplest case, we can represent it like this

$$(A_N + I) (A_{N-1} + I) ... (A_1 + I) x$$

where $$I$$ is the identity matrix.
This identity matrix acts precisely in the same way as the residual connection - it allows
the input $$x$$ to propagate deeper to the network.

Such representation shows that a residual network initialized with random numbers is lies
father from the zero point.
And this is beneficial as the coordinate center is a saddle point.

A different paper
[(Ge et al., 2015)](https://arxiv.org/abs/1503.02101)
says that the stochasticity of the stochastic gradient descent (SGD) may be
the reason we can perform efficient (polynomial in time) nonconvex optimization.
Imagine you are stuck at a saddle point, your gradient is zero.
Does this mean you won’t move?
It does if you would know the exact gradient, but in practice we estimate it using
stochastic minibatch approximation.
So your gradient estimate would likely have some noise and would lead you in some
random direction, further from the saddle point if you are lucky.
And if the saddle points are not stable, you don’t need much luck.
Their actual contribution is proof that for a broad set of nonconvex functions
(strict saddles) saddle points are indeed unstable and a slight modification of SGD
allows the optimizer to avoid saddles and to converge in polynomial time with a high
probability.

Besides, nonstochastic gradient descent can take an exponentially long time
to escape saddle points
[(Du et al. ,2017)](https://arxiv.org/abs/1705.10412).
This is interesting to see how an engineering solution to a gradient estimation problem
instead of making the optimization worse (you don’t get the true gradient anymore)
actually led us to qualitatively better optimizers.

At this point, we need to understand that as we go deeper into the worlds of neural
networks we dive into the world of nuances and assumptions.
On one hand, we have SGD where stochasticity is the primary property that allows
avoiding being stuck at the saddle points.
On the other hand - this result may be too general or even too specific to hold true
for neural networks, given enough assumptions at least.

However, more papers come in on stage
(
[Allen-Zhu et al.](https://arxiv.org/abs/1811.03962)
,
[2019 and Du et al., 2019](https://arxiv.org/abs/1811.03804)
).
The first of them proves that if the dataset is not contradictory and your ReLu-network
(FC or CNN or residual) is big enough (polynomial in dataset size), for a common
initialization scheme with a high probability both SG and SGD find a minimum with 100%
training accuracy for a classification task or epsilon-error global minimum for a
regression task in polynomial time for a small enough learning rate.
The second paper shows a similar result specifically for the
least-squares loss, but for a broader range of activation functions.

This may look like a lot of prerequisites, but this is exactly what we observe in practice.
1. Non-contradictory dataset - you won’t get 100% accuracy if you have the same image
labeled both as “dog” and “cat”.
You either predict “dog” or “cat” and get the error on the other
training point.
1. Very small networks are surprisingly hard to train in practice - set all your hidden
sizes to 10 and see what happens.
1. Initialization matters, and we will talk more about it in future posts
1. The learning rate also matters; you can hope to get good enough results for
ADAM(lr=1e-3), but a vanilla SGD is a tricky beast.

## ~~Stack more layers!~~ Add more parameters!

There’s still a gap between a theoretical understanding of neural network trainability
and practice. “Polynomially large neural network” sounds good for complexity theory
people, but what if the polynomial is of the order 60?
Many papers have been trying to decrease the degree,
[Kawaguchi and Huang (2019)](https://arxiv.org/abs/1908.02419)
summarize the attempts in the table below.

<figure>
   <img src="{{'/assets/images/theory1_trainability.png' | relative_url }}"> 
   <figcaption>Table 1. Number of parameters required to ensure the trainability,
   in terms of n, where n is the number of samples in a training dataset and H is the
   number of hidden layers.
   (<a href="https://arxiv.org/abs/1908.02419">Kawaguchi and Huang, 2019</a>)
   </figcaption>
</figure>

They also demonstrate a common practice example -- a relatively small ResNet-18 can
achieve 100% accuracy on a bunch of datasets including a dataset of 50 000 random 3x24x24
images with random 0-9 labels.

<figure>
   <img src="{{'/assets/images/theory1_params_exp.png' | relative_url }}"> 
   <figcaption>Fig. 4. ResNet-18 succesfully fits several computer vision datasets
   includng a randomly generated one of size 50 000.
   (<a href="https://arxiv.org/abs/1908.02419">Kawaguchi and Huang, 2019</a>)
   </figcaption>
</figure>

The assumptions are:
1. The loss has Lipschitz (smoothish) gradient
2. The activation function is analytic, Lipschitz, monotonically increasing and has the limits at both ends

The squared loss and cross-entropy loss both satisfy the first assumption and
Sigmoid and Softplus satisfy the second.
The authors also note that Softplus ($$ln(1 + exp(ax))/a$$) can approximate ReLu for any
desired accuracy while satisfying the assumption 2.

It should be indicated, however, that this particular paper enforces zero learning rate
on all layers except the last one as
[Das and Golikov, (2019)](https://arxiv.org/pdf/1911.05402.pdf)
mention.
This assumption is not practical and it is possible that we’re yet to see a true
asymptotic of a neural network size required for trainability.

A new experiment-inspired wave of theory papers has answered a lot of questions about
the trainability of neural networks using gradient descent methods.
We know that it is the structure of the neural networks that makes all minima
(at least roughly) the same and that it is the number of parameters that makes it
possible to use simple gradient descent-based methods for efficient (polynomial-time)
optimization.
There still exist questions about the assumptions taken and about the particular
asymptotics, but we’re on the right track.
