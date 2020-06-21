# What Deep Learning Theories Can Tell Us Today

Deep learning has been with us for a while now and mostly it’s been studied in a very practice-first way.
And this way has its pros, a lot of them.
Deep learning changed how people communicate using different languages by greatly improving machine translation quality.
It is an essential part of modern dialogue systems such as Siri, Alexa, and Google Assistant.
And you can do a bunch of cool stuff with images and videos including deoldifying, enhancing the quality and so on.

But if we are talking science here, there are many reasons to deviate from such an approach and to try to
understand why our practical results are so good.
Because classic ML and basic reasoning tell us that modern neural networks should not be useful at all.

In this small series of posts we will talk about the state of modern deep learning theories that try to explain deep learning
success and to get some new insights about it.
We will mostly discuss the results omiting the proofs and precice theorem formulations.
This series is written by practice people who are interested in what is happening in theory and 
we would not claim that it reviews most of the key theoretical works,
but we like to mention some of the results we personally find interesting and important.

We want to start with the question of why we are able to optimize neural networks using simple gradient descent methods
and not to be stuck in some bad local minimum.

## How bad is this local minimum?

Let’s look at a picture you probably saw in your university classes.
It is about gradient descent and convex optimization.

<figure>
	<img src="{{'/assets/images/local-minimum.png' | relative_url }}"> 
	<figcaption>Fig. 1. Gradient descent stuck at local minima. [source](https://www.researchgate.net/figure/Gradient-Descent-Stuck-at-Local-Minima-18_fig4_338621083)
	</figcaption>
</figure>

Gradient descent is greedy.
It just improves your loss locally and it is commonly accepted that GS is guaranteed to converge to a global minimum only in a convex setup.
On the other hand, the neural network loss landscape is highly complex and nonconvex.

<figure>
	<img src="{{'/assets/images/loss-landscape.png' | relative_url }}"> 
	<figcaption>Visualizing and Understanding the Effectiveness of BERT, Hao et al. ([arxiv](https://arxiv.org/abs/1908.05620))
	</figcaption>
</figure>

Several researchers who did neural networks before the era of deep learning, including Yann LeCun,
[say](https://lexfridman.com/yann-lecun) that the fact that NNs can be optimized using gradient descent fascinates
them and people would not expect it to work back in the days.

But it does.
And why is not the question of applied DL, but is the question of theory.
Take linear (or logistic) regression which is the simplest kind of neural network one can find.
Linear regression loss is just (y - Ax)^2 which is simple and convex and only has one local minimum - its global minimum.
[Lu and Kawaguchi (2017)](https://arxiv.org/abs/1702.08580)
studied a bit more complex form of a neural network.
Linear neural network - one that does not have nonlinear activation functions - $A_N A_N-1 … A_1 x$.
It is easy to see that such a network, just like linear regression, can only describe linear functions.
But the optimization problem is not trivial and the loss-landscape is not convex.
Their result is quite fascinating.
These networks have multiple local minima, but they are all the same.
So even if you always go the steepest curve you find the lowest valley!

A similar result has been known for some time in wide (enough) shallow networks with sigmoid nonlinearities
[[Yu and Chen, 1995](https://arxiv.org/abs/1704.08045)]
and
[Nguyen and Hein (2017)](https://arxiv.org/abs/1704.08045)
generalized it to deep sigmoid networks (with the layer size shrinking down after some point).

And recent results go much further beyond that.
Kawaguchi and Kaelbling (2019) show that adding a single neuron to each output unit of an arbitrary neural network
(e.g., any fully-connected or convolutional neural network with any depth and any width, with or without skip connections)
actually leads to the elimination of all local minima (they become global).
More precisely:

For any neural network $f’$
$$
f’ (x; θ) = f (x; θ) + g(x; a, b, W )
$$
$$
g(x;a,b,W)k = ak exp(wk⊤x + bk)
$$

and a differentiable and convex loss function (e.g., squared loss, cross-entropy loss, polynomial hinge loss, …)
regularized via L2 norm of vector $a$ every local minimum is global.

Now let’s go several steps back and think about this.
The results in these papers are quite amazing, but how much are they applicable to the real world?
We don’t use linear and sigmoid networks in practice (we hope, you too) and we don’t add any special
neurons to simplify the loss landscape (ok, we actually do, but BatchNorm will be discussed later).
But the progress the theoretical community made in the last few years is tremendous and we will
probably see even more realistic setups this year.

However, the most important part is that even if we have only global minima,
this does not mean that gradient descent can find even one of them in a reasonable time.
Imagine an extreme case where you have a huge (more than your learning rate) and a completely flat plateau.
As soon as your optimizer leads you there, you are stuck forever.

<figure>
	<img src="{{'/assets/images/plateau.png' | relative_url }}"> 
	<figcaption> Plateau
	</figcaption>
</figure>


## Where do you lead me (S)GD?

So the landscape of NNs should be even more special.
For example, not only linear networks do not have bad local minima, but also if you reparametrize them as a residual network

$$
A_i^\hat = A_i + I
$$

any critical point is a global minimum
[[Hardt and Ma, 2016](https://arxiv.org/abs/1611.04231)]
i.e. you can avoid saddle points via a simple resnet reparametrization.

Now we step on the field of training dynamics, which is not solely defined by the loss landscape but also may depend on the initialization and some luck.
[Ge et al. (2015)](https://arxiv.org/abs/1503.02101)
say that the stochasticity of the stochastic gradient descent (SGD) may be exactly the reason we are able to perform efficient
(polynomial in time) nonconvex optimization.
Imagine you are stuck at a saddle point, your gradient is zero.
Does this mean you won’t move?
It does if you would know the exact gradient (0), but in practice, our datasets and networks are big enough
that we need to use a minibatch stochastic approximation of the real gradient.
So your gradient estimate would likely have some noise and would lead you in some random direction,
further from the saddle point if you are lucky.
And if the saddle points are not stable, you don’t need much luck.
Their actual contribution is proof that for a broad set of nonconvex functions (strict saddles) saddle points
are indeed unstable and a slight modification of SGD allows you to avoid the saddles and to converge in polynomial time with a high probability.


<figure>
	<img src="{{'/assets/images/ge_et_al.png' | relative_url }}"> 
	<figcaption> Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition, Ge et al. ([arxiv](https://arxiv.org/abs/1503.02101)
	</figcaption>
</figure>

On the other hand, nonstochastic gradient descent can take an exponentially long time to escape saddle points
[[Du et al. ,2017](https://arxiv.org/abs/1705.10412)].
This is interesting to see how an engineering solution to a gradient estimation problem instead of making the
optimization worse (you don’t get the true gradient anymore) actually led us to qualitatively better optimizers.


<figure>
	<img src="{{'/assets/images/du_et_al.png' | relative_url }}"> 
	<figcaption> Gradient Descent Can Take Exponential Time to Escape Saddle Points, Du et al. ([arxiv](https://arxiv.org/abs/1705.10412)
	</figcaption>
</figure>


<figure>
	<img src="{{'/assets/images/du_et_al_image.png' | relative_url }}"> 
	<figcaption> Gradient Descent Can Take Exponential Time to Escape Saddle Points, Du et al. ([arxiv](https://arxiv.org/abs/1705.10412)
	</figcaption>
</figure>

This is a point when we step even deeper into the world of nuances and assumptions.
On the one hand we have SGD where stochasticity is the primary property that allows avoiding being stuck at the saddle points.
On the other hand - this result may be too general or even too specific to hold true for neural networks, given enough assumptions at least.

However, more papers come in on stage
[[Allen-Zhu et al, 2019](https://arxiv.org/abs/1811.03962) and [Du et al., 2019](https://arxiv.org/abs/1811.03804)].
The first of them proves that if the dataset is not contradictory and your ReLu-network (FC or CNN or residual) is big enough,
for a common initialization scheme with a high probability both SG and SGD find a minimum with 100% training accuracy for a
classification task or epsilon-error global minimum for a regression task in polynomial time for a small enough learning rate.
The second paper shows a similar result specifically for the least-squares loss, but for a broader range of activation functions
(which excludes ReLu but includes Sigmoid and SoftPlus).

This may look like a lot of prerequisites, but this is exactly what we observe in practice.
Non-contradictory dataset - you won’t get 100% accuracy if you have the same image labeled both as “dog” and “cat”. You either predict “dog” or “cat” and get the error on the other training point.
Very small networks are surprisingly hard to train - set all your hidden sizes to 10 and see what happens.
Initialization matters, and we will talk more about it
The learning rate also matters; you can hope to get good enough results for ADAM(lr=1e-3), but a vanilla SGD is a tricky beast.

## ~~Stack more layers!~~ Add more parameters!

There’s still a gap between a theoretical understanding of neural network trainability and practice.
“Polynomially large neural network” sounds good for complexity theory people, but what if the polynomial is of the order 60?
Many papers have been trying to decrease the degree,
[Kawaguchi and Huang (2019)](https://arxiv.org/abs/1908.02419)
summarize the attempts in the table below

<figure>
	<img src="{{'/assets/images/kawaguchi_and_huang_table.png' | relative_url }}"> 
	<figcaption> Gradient Descent Finds Global Minima for Generalizable Deep Neural Networks of Practical Sizes, Kawaguchi and Huang. ([arxiv](https://arxiv.org/abs/1908.02419)
	</figcaption>
</figure>

and also demonstrate a common practice example -- a relatively small ResNet-18 can achieve 100% accuracy on a bunch of
datasets including a dataset of 50 000 random 3x24x24 images with random 0-9 labels. 

<figure>
	<img src="{{'/assets/images/kawaguchi_and_huang_plot.png' | relative_url }}"> 
	<figcaption> Gradient Descent Finds Global Minima for Generalizable Deep Neural Networks of Practical Sizes, Kawaguchi and Huang. ([arxiv](https://arxiv.org/abs/1908.02419)
	</figcaption>
</figure>

The previous theories require about n^8 = (50 000)^8 ~= 10^37 parameters to do that,
ResNet-18 has about 10^7 parameters which is a significantly smaller number.
[Kawaguchi and Huang (2019)](https://arxiv.org/abs/1908.02419)
theory require just n^1 parameters.

The assumptions are:
  1. the loss has Lipschitz (smoothish) gradient
  2. the activation function is analytic, Lipschitz, monotonically increasing and has the limits at both ends

The squared loss and cross-entropy loss both satisfy the first assumption and Sigmoid and Softplus satisfy the second.
The authors also note that Softplus ($ln(1 + exp(ax))/a$) can approximate ReLu for any desired accuracy while satisfying the assumption 2.

A new experiment-inspired wave of theory papers has answered a lot of questions about the trainability of neural networks
using gradient descent methods. We know that it is the structure of the neural networks that makes all minima (at least roughly)
the same and that it is the number of parameters that makes it possible to use simple gradient descent-based methods for efficient
(polynomial-time) optimization.
There still exist questions about the assumptions taken and about the particular asymptotics, but we’re on the right track.
