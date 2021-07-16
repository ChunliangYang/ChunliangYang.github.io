Hi, I am now a first-year master student at Texas A&M University (<b>TAMU-ISE</b>).

<hr>


## <i class="iconfont icon-news" style="font-size: 0.9em"></i> News
<table class="table table-hover">
<tr>
  <td class='col-md-3'>2021-07-14</td>
 <td>One paper about deep level set was accepted by ICCAD 2021.</td>
</tr>
</table>

 <!--

## <i class="iconfont icon-education" style="font-size: 0.9em"></i> Education

<table class="table table-hover">
  <tr>
    <td class="col-md-3">Aug 2021 - Present</td>
    <td>
        <strong>M.S. in Industrial Engineering</strong>
        <br>
      Texas A&M University
    </td>
  </tr>
  <tr>
    <td class="col-md-3">Sep 2017 - June 2021</td>
    <td>
        <strong>B.S. in Industrial Engineering</strong>
        <br>
        Beijing Jiaotong University
    </td>
  </tr>
  <tr>
  </tr>
  <!-- <tr>
    <td class="col-md-3">Aug 2012 - May 2015</td>
    <td>
      The fifth high school of Xiangyang (Hubei, China)
    </td>
  </tr> -->
</table>

-->

## <i class="iconfont icon-gongzuojingyan" style="font-size: 0.9em"></i> Experience
<table class="table table-hover">
<tr>
  <td class='col-md-3'>Nov 2020 - July 2021</td>
  <td><strong>Smartmore Co.Ltd</strong>, Research Intern</td>
</tr>
<tr>
  <td class='col-md-3'>Aug 2020 - Aug 2021</td>
  <td><strong>CUHK</strong>, Research Assistant</td>
</tr>
<tr>
  <td class='col-md-3'>June 2018 - Sept 2018</td>
  <td><strong>Tencent Labs</strong>, Research Intern</td>
</tr>
</table>

<!--
## <i class="fa fa-chevron-right"></i> Selected Publications <a href="https://github.com/bamos/cv/blob/master/publications/selected.bib"><i class="fa fa-code-fork" aria-hidden="true"></i></a>

<a href="https://scholar.google.com/citations?user=d8gdZR4AAAAJ" class="btn btn-primary" style="padding: 0.3em;">
  <i class="ai ai-google-scholar"></i> Google Scholar
</a>

<table class="table table-hover">
<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1909.12830' target='_blank'><img src="images/publications/amos2020differentiable.png"/></a> </td>
<td>
    <strong>The Differentiable Cross-Entropy Method</strong><br>
    <strong>B. Amos</strong> and D. Yarats<br>
    ICML 2020<br>

    [1]
[<a href='javascript:;'
    onclick='$("#abs_amos2020differentiable").toggle()'>abs</a>] [<a href='https://arxiv.org/abs/1909.12830' target='_blank'>pdf</a>] <br>

<div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1">
We study the Cross-Entropy Method (CEM) for the non-convex
optimization of a continuous and parameterized
objective function and introduce a differentiable
variant (DCEM) that enables us to differentiate the
output of CEM with respect to the objective
function's parameters. In the machine learning
setting this brings CEM inside of the end-to-end
learning pipeline where this has otherwise been
impossible. We show applications in a synthetic
energy-based structured prediction task and in
non-convex continuous control. In the control
setting we show on the simulated cheetah and walker
tasks that we can embed their optimal action
sequences with DCEM and then use policy optimization
to fine-tune components of the controller as a step
towards combining model-based and model-free RL.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/2002.04523' target='_blank'><img src="images/publications/lambert2020objective.png"/></a> </td>
<td>
    <strong>Objective Mismatch in Model-based Reinforcement Learning</strong><br>
    N. Lambert, <strong>B. Amos</strong>, O. Yadan, and R. Calandra<br>
    L4DC 2020<br>

    [2]
[<a href='javascript:;'
    onclick='$("#abs_lambert2020objective").toggle()'>abs</a>] [<a href='https://arxiv.org/abs/2002.04523' target='_blank'>pdf</a>] <br>

<div id="abs_lambert2020objective" style="text-align: justify; display: none" markdown="1">
Model-based reinforcement learning (MBRL) has been shown to be a powerful framework for data-efficiently learning control of continuous tasks. Recent work in MBRL has mostly focused on using more advanced function approximators and planning schemes, with little development of the general framework. In this paper, we identify a fundamental issue of the standard MBRL framework-what we call the objective mismatch issue. Objective mismatch arises when one objective is optimized in the hope that a second, often uncorrelated, metric will also be optimized. In the context of MBRL, we characterize the objective mismatch between training the forward dynamics model wrt the likelihood of the one-step ahead prediction, and the overall goal of improving performance on a downstream control task. For example, this issue can emerge with the realization that dynamics models effective for a specific task do not necessarily need to be globally accurate, and vice versa globally accurate models might not be sufficiently accurate locally to obtain good control performance on a specific task. In our experiments, we study this objective mismatch issue and demonstrate that the likelihood of one-step ahead predictions is not always correlated with control performance. This observation highlights a critical limitation in the MBRL framework which will require further research to be fully understood and addressed. We propose an initial method to mitigate the mismatch issue by re-weighting dynamics model training. Building on it, we conclude with a discussion about other potential directions of research for addressing this issue.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf' target='_blank'><img src="images/publications/amos2019differentiable3.png"/></a> </td>
<td>
    <strong>Differentiable Convex Optimization Layers</strong><br>
    A. Agrawal*, <strong>B. Amos*</strong>, S. Barratt*, S. Boyd*, S. Diamond*, and J. Z. Kolter*<br>
    NeurIPS 2019<br>

    [3]
[<a href='javascript:;'
    onclick='$("#abs_amos2019differentiable3").toggle()'>abs</a>] [<a href='http://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf' target='_blank'>pdf</a>]  [<a href='https://github.com/cvxgrp/cvxpylayers' target='_blank'>code</a>] <br>

<div id="abs_amos2019differentiable3" style="text-align: justify; display: none" markdown="1">
Recent work has shown how to embed differentiable optimization problems (that is, problems whose solutions can be backpropagated through) as layers within deep learning architectures. This method provides a useful inductive bias for certain problems, but existing software for differentiable optimization layers is rigid and difficult to apply to new settings. In this paper, we propose an approach to differentiating through disciplined convex programs, a subclass of convex optimization problems used by domain-specific languages (DSLs) for convex optimization. We introduce disciplined parametrized programming, a subset of disciplined convex programming, and we show that every disciplined parametrized program can be represented as the composition of an affine map from parameters to problem data, a solver, and an affine map from the solver’s solution to a solution of the original problem (a new form we refer to as affine-solver-affine form). We then demonstrate how to efficiently differentiate through each of these components, allowing for end-to-end analytical differentiation through the entire convex program. We implement our methodology in version 1.1 of CVXPY, a popular Python-embedded DSL for convex optimization, and additionally implement differentiable layers for disciplined convex programs in PyTorch and TensorFlow 2.0. Our implementation significantly lowers the barrier to using convex optimization problems in differentiable programs. We present applications in linear machine learning models and in stochastic control, and we show that our layer is competitive (in execution time) compared to specialized differentiable solvers from past work.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1910.01727' target='_blank'><img src="images/publications/grefenstette2019generalized.png"/></a> </td>
<td>
    <strong>Generalized Inner Loop Meta-Learning</strong><br>
    E. Grefenstette, <strong>B. Amos</strong>, D. Yarats, P. Htut, A. Molchanov, F. Meier, D. Kiela, K. Cho, and S. Chintala<br>
    arXiv 2019<br>

    [4]
[<a href='javascript:;'
    onclick='$("#abs_grefenstette2019generalized").toggle()'>abs</a>] [<a href='https://arxiv.org/abs/1910.01727' target='_blank'>pdf</a>]  [<a href='https://github.com/facebookresearch/higher' target='_blank'>code</a>] <br>

<div id="abs_grefenstette2019generalized" style="text-align: justify; display: none" markdown="1">
Many (but not all) approaches self-qualifying as "meta-learning" in
deep learning and reinforcement learning fit a
common pattern of approximating the solution to a
nested optimization problem. In this paper, we give
a formalization of this shared pattern, which we
call GIMLI, prove its general requirements, and
derive a general-purpose algorithm for implementing
similar approaches. Based on this analysis and
algorithm, we describe a library of our design, higher, which we share with the community to assist
and enable future research into these kinds of
meta-learning approaches. We end the paper by
showcasing the practical applications of this
framework and library through illustrative
experiments and ablation studies which they
facilitate.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1906.08707' target='_blank'><img src="images/publications/amos2019limited.png"/></a> </td>
<td>
    <strong>The Limited Multi-Label Projection Layer</strong><br>
    <strong>B. Amos</strong>, V. Koltun, and J. Z. Kolter<br>
    arXiv 2019<br>

    [5]
[<a href='javascript:;'
    onclick='$("#abs_amos2019limited").toggle()'>abs</a>] [<a href='https://arxiv.org/abs/1906.08707' target='_blank'>pdf</a>]  [<a href='https://github.com/locuslab/lml' target='_blank'>code</a>] <br>

<div id="abs_amos2019limited" style="text-align: justify; display: none" markdown="1">
We propose the Limited Multi-Label (LML) projection layer as a new
primitive operation for end-to-end learning systems. The LML layer
provides a probabilistic way of modeling multi-label predictions
limited to having exactly k labels. We derive efficient forward and
backward passes for this layer and show how the layer can be used to
optimize the top-k recall for multi-label tasks with incomplete label
information. We evaluate LML layers on top-k CIFAR-100 classification
and scene graph generation. We show that LML layers add a negligible
amount of computational overhead, strictly improve the model's
representational capacity, and improve accuracy. We also revisit the
truncated top-k entropy method as a competitive baseline for top-k
classification.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://github.com/bamos/thesis/raw/master/bamos_thesis.pdf' target='_blank'><img src="images/publications/amos2019differentiable.png"/></a> </td>
<td>
    <strong>Differentiable Optimization-Based Modeling for Machine Learning</strong><br>
    <strong>B. Amos</strong><br>
    Ph.D. Thesis 2019<br>

    [6] [<a href='https://github.com/bamos/thesis/raw/master/bamos_thesis.pdf' target='_blank'>pdf</a>]  [<a href='https://github.com/bamos/thesis' target='_blank'>code</a>] <br>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1810.13400' target='_blank'><img src="images/publications/amos2018end.png"/></a> </td>
<td>
    <strong>Differentiable MPC for End-to-end Planning and Control</strong><br>
    <strong>B. Amos</strong>, I. Rodriguez, J. Sacks, B. Boots, and J. Z. Kolter<br>
    NeurIPS 2018<br>

    [7]
[<a href='javascript:;'
    onclick='$("#abs_amos2018end").toggle()'>abs</a>] [<a href='https://arxiv.org/abs/1810.13400' target='_blank'>pdf</a>]  [<a href='https://locuslab.github.io/mpc.pytorch/' target='_blank'>code</a>] <br>

<div id="abs_amos2018end" style="text-align: justify; display: none" markdown="1">
We present foundations for using Model Predictive Control (MPC) as a differentiable policy class for reinforcement learning in continuous state and action spaces. This provides one way of leveraging and combining the advantages of model-free and model-based approaches. Specifically, we differentiate through MPC by using the KKT conditions of the convex approximation at a fixed point of the controller. Using this strategy, we are able to learn the cost and dynamics of a controller via end-to-end learning. Our experiments focus on imitation learning in the pendulum and cartpole domains, where we learn the cost and dynamics terms of an MPC policy class. We show that our MPC policies are significantly more data-efficient than a generic neural network and that our method is superior to traditional system identification in a setting where the expert is unrealizable.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://arxiv.org/abs/1805.08195' target='_blank'><img src="images/publications/brown2018depth.png"/></a> </td>
<td>
    <strong>Depth-Limited Solving for Imperfect-Information Games</strong><br>
    N. Brown, T. Sandholm, and <strong>B. Amos</strong><br>
    NeurIPS 2018<br>

    [8]
[<a href='javascript:;'
    onclick='$("#abs_brown2018depth").toggle()'>abs</a>] [<a href='http://arxiv.org/abs/1805.08195' target='_blank'>pdf</a>] <br>

<div id="abs_brown2018depth" style="text-align: justify; display: none" markdown="1">
A fundamental challenge in imperfect-information games is that states do not have well-defined values. As a result, depth-limited search algorithms used in single-agent settings and perfect-information games do not apply. This paper introduces a principled way to conduct depth-limited solving in imperfect-information games by allowing the opponent to choose among a number of strategies for the remainder of the game at the depth limit. Each one of these strategies results in a different set of values for leaf nodes. This forces an agent to be robust to the different strategies an opponent may employ. We demonstrate the effectiveness of this approach by building a master-level heads-up no-limit Texas hold'em poker AI that defeats two prior top agents using only a 4-core CPU and 16 GB of memory. Developing such a powerful agent would have previously required a supercomputer.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://openreview.net/forum?id=r1HhRfWRZ' target='_blank'><img src="images/publications/amos2018learning.png"/></a> </td>
<td>
    <strong>Learning Awareness Models</strong><br>
    <strong>B. Amos</strong>, L. Dinh, S. Cabi, T. Roth&ouml;rl, S. Colmenarejo, A. Muldal, T. Erez, Y. Tassa, N. de Freitas, and M. Denil<br>
    ICLR 2018<br>

    [9]
[<a href='javascript:;'
    onclick='$("#abs_amos2018learning").toggle()'>abs</a>] [<a href='https://openreview.net/forum?id=r1HhRfWRZ' target='_blank'>pdf</a>] <br>

<div id="abs_amos2018learning" style="text-align: justify; display: none" markdown="1">
We consider the setting of an agent with a fixed body interacting with an
unknown and uncertain external world. We show that models
trained to predict proprioceptive information about the
agent's body come to represent objects in the external world.
In spite of being trained with only internally available
signals, these dynamic body models come to represent external
objects through the necessity of predicting their effects on
the agent's own body. That is, the model learns holistic
persistent representations of objects in the world, even
though the only training signals are body signals. Our
dynamics model is able to successfully predict distributions
over 132 sensor readings over 100 steps into the future and we
demonstrate that even when the body is no longer in contact
with an object, the latent variables of the dynamics model
continue to represent its shape. We show that active data
collection by maximizing the entropy of predictions about the
body-touch sensors, proprioception and vestibular
information-leads to learning of dynamic models that show
superior performance when used for control. We also collect
data from a real robotic hand and show that the same models
can be used to answer questions about properties of objects in
the real world. Videos with qualitative results of our models
are available <a href="https://goo.gl/mZuqAV">here</a>.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://arxiv.org/abs/1703.04529' target='_blank'><img src="images/publications/donti2017task.png"/></a> </td>
<td>
    <strong>Task-based End-to-end Model Learning</strong><br>
    P. Donti, <strong>B. Amos</strong>, and J. Z. Kolter<br>
    NeurIPS 2017<br>

    [10]
[<a href='javascript:;'
    onclick='$("#abs_donti2017task").toggle()'>abs</a>] [<a href='http://arxiv.org/abs/1703.04529' target='_blank'>pdf</a>]  [<a href='https://github.com/locuslab/e2e-model-learning' target='_blank'>code</a>] <br>

<div id="abs_donti2017task" style="text-align: justify; display: none" markdown="1">
As machine learning techniques have become more ubiquitous, it has
become common to see machine learning prediction algorithms operating
within some larger process. However, the criteria by which we train
machine learning algorithms often differ from the ultimate criteria on
which we evaluate them. This paper proposes an end-to-end approach for
learning probabilistic machine learning models within the context of
stochastic programming, in a manner that directly captures the
ultimate task-based objective for which they will be used. We then
present two experimental evaluations of the proposed approach, one as
applied to a generic inventory stock problem and the second to a
real-world electrical grid scheduling task. In both cases, we show
that the proposed approach can outperform both a traditional modeling
approach and a purely black-box policy optimization approach.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://arxiv.org/abs/1703.00443' target='_blank'><img src="images/publications/amos2017optnet.png"/></a> </td>
<td>
    <strong>OptNet: Differentiable Optimization as a Layer in Neural Networks</strong><br>
    <strong>B. Amos</strong> and J. Z. Kolter<br>
    ICML 2017<br>

    [11]
[<a href='javascript:;'
    onclick='$("#abs_amos2017optnet").toggle()'>abs</a>] [<a href='http://arxiv.org/abs/1703.00443' target='_blank'>pdf</a>]  [<a href='https://github.com/locuslab/optnet' target='_blank'>code</a>] <br>

<div id="abs_amos2017optnet" style="text-align: justify; display: none" markdown="1">
This paper presents OptNet, a network architecture that integrates
optimization problems (here, specifically in the form of quadratic programs)
as individual layers in larger end-to-end trainable deep networks.
These layers encode constraints and complex dependencies
between the hidden states that traditional convolutional and
fully-connected layers often cannot capture.
In this paper, we explore the foundations for such an architecture:
we show how techniques from sensitivity analysis, bilevel
optimization, and implicit differentiation can be used to
exactly differentiate through these layers and with respect
to layer parameters;
we develop a highly efficient solver for these layers that exploits fast
GPU-based batch solves within a primal-dual interior point method, and which
provides backpropagation gradients with virtually no additional cost on top of
the solve;
and we highlight the application of these approaches in several problems.
In one notable example, we show that the method is
capable of learning to play mini-Sudoku (4x4) given just input and output games, with no a priori information about the rules of the game;
this highlights the ability of our architecture to learn hard
constraints better than other neural architectures.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://arxiv.org/abs/1609.07152' target='_blank'><img src="images/publications/amos2017input.png"/></a> </td>
<td>
    <strong>Input Convex Neural Networks</strong><br>
    <strong>B. Amos</strong>, L. Xu, and J. Z. Kolter<br>
    ICML 2017<br>

    [12]
[<a href='javascript:;'
    onclick='$("#abs_amos2017input").toggle()'>abs</a>] [<a href='http://arxiv.org/abs/1609.07152' target='_blank'>pdf</a>]  [<a href='https://github.com/locuslab/icnn' target='_blank'>code</a>] <br>

<div id="abs_amos2017input" style="text-align: justify; display: none" markdown="1">
This paper presents the input convex neural network
architecture. These are scalar-valued (potentially deep) neural
networks with constraints on the network parameters such that the
output of the network is a convex function of (some of) the inputs.
The networks allow for efficient inference via optimization over some
inputs to the network given others, and can be applied to settings
including structured prediction, data imputation, reinforcement
learning, and others. In this paper we lay the basic groundwork for
these models, proposing methods for inference, optimization and
learning, and analyze their representational power. We show that many
existing neural network architectures can be made input-convex with
a minor modification, and develop specialized optimization
algorithms tailored to this setting. Finally, we highlight the
performance of the methods on multi-label prediction, image
completion, and reinforcement learning problems, where we show
improvement over the existing state of the art in many cases.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://proceedings.mlr.press/v48/zhaoa16.html' target='_blank'><img src="images/publications/zhao2016collapsed.png"/></a> </td>
<td>
    <strong>Collapsed Variational Inference for Sum-Product Networks</strong><br>
    H. Zhao, T. Adel, G. Gordon, and <strong>B. Amos</strong><br>
    ICML 2016<br>

    [13]
[<a href='javascript:;'
    onclick='$("#abs_zhao2016collapsed").toggle()'>abs</a>] [<a href='http://proceedings.mlr.press/v48/zhaoa16.html' target='_blank'>pdf</a>] <br>

<div id="abs_zhao2016collapsed" style="text-align: justify; display: none" markdown="1">
Sum-Product Networks (SPNs) are probabilistic inference machines that admit
exact inference in linear time in the size of the network. Existing
parameter learning approaches for SPNs are largely based on the maximum
likelihood principle and hence are subject to overfitting compared to
more Bayesian approaches. Exact Bayesian posterior inference for SPNs is
computationally intractable. Both standard variational inference and
posterior sampling for SPNs are computationally infeasible even for
networks of moderate size due to the large number of local latent
variables per instance. In this work, we propose a novel deterministic
collapsed variational inference algorithm for SPNs that is
computationally efficient, easy to implement and at the same time allows
us to incorporate prior information into the optimization formulation.
Extensive experiments show a significant improvement in accuracy compared
with a maximum likelihood based approach.
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://reports-archive.adm.cs.cmu.edu/anon/anon/2016/CMU-CS-16-118.pdf' target='_blank'><img src="images/publications/amos2016openface.png"/></a> </td>
<td>
    <strong>OpenFace: A general-purpose face recognition library with mobile applications</strong><br>
    <strong>B. Amos</strong>, B. Ludwiczuk, and M. Satyanarayanan<br>
    CMU 2016<br>

    [14]
[<a href='javascript:;'
    onclick='$("#abs_amos2016openface").toggle()'>abs</a>] [<a href='http://reports-archive.adm.cs.cmu.edu/anon/anon/2016/CMU-CS-16-118.pdf' target='_blank'>pdf</a>]  [<a href='https://cmusatyalab.github.io/openface' target='_blank'>code</a>] <br>

<div id="abs_amos2016openface" style="text-align: justify; display: none" markdown="1">
Cameras are becoming ubiquitous in the Internet of Things (IoT) and
can use face recognition technology to improve context. There is a
large accuracy gap between today's publicly available face recognition
systems and the state-of-the-art private face recognition
systems. This paper presents our OpenFace face recognition library
that bridges this accuracy gap. We show that OpenFace provides
near-human accuracy on the LFW benchmark and present a new
classification benchmark for mobile scenarios. This paper is intended
for non-experts interested in using OpenFace and provides a light
introduction to the deep neural network techniques we use.

We released OpenFace in October 2015 as an open source library under
the Apache 2.0 license. It is available at:
<http://cmusatyalab.github.io/openface/>
</div>

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://dl.acm.org/doi/10.1145/3374219' target='_blank'><img src="images/publications/amos2014QNSTOP.png"/></a> </td>
<td>
    <strong>QNSTOP: Quasi-Newton Algorithm for Stochastic Optimization</strong><br>
    <strong>B. Amos</strong>, D. Easterling, L. Watson, W. Thacker, B. Castle, and M. Trosset<br>
    ACM TOMS 2014<br>

    [15]
[<a href='javascript:;'
    onclick='$("#abs_amos2014QNSTOP").toggle()'>abs</a>] [<a href='https://dl.acm.org/doi/10.1145/3374219' target='_blank'>pdf</a>] <br>

<div id="abs_amos2014QNSTOP" style="text-align: justify; display: none" markdown="1">
QNSTOP consists of serial and parallel (OpenMP) Fortran 2003 codes for the
quasi-Newton stochastic optimization method of Castle and Trosset. For
stochastic problems, convergence theory exists for the particular
algorithmic choices and parameter values used in QNSTOP. Both the parallel
driver subroutine, which offers several parallel decomposition strategies, and the serial driver subroutine can be used for stochastic optimization or
deterministic global optimization, based on an input switch. QNSTOP is
particularly effective for “noisy” deterministic problems, using only
objective function values. Some performance data for computational systems
biology problems is given.
</div>

</td>
</tr>

</table> -->


<!-- ## <i class="fa fa-chevron-right"></i> Teaching Experience
<table class="table table-hover">
<tr>
  <td class='col-md-1'>S2017</td>
  <td><strong>Graduate AI</strong> (CMU 15-780), TA</td>
</tr>
<tr>
  <td class='col-md-1'>S2016</td>
  <td><strong>Distributed Systems</strong> (CMU 15-440/640), TA</td>
</tr>
<tr>
  <td class='col-md-1'>S2013</td>
  <td><strong>Software Design and Data Structures</strong> (VT CS 2114), TA</td>
</tr>
</table> -->



<!--
## <i class="fa fa-chevron-right"></i> Service
<table class="table table-hover">
<tr>
  <td class='col-md-2'>Reviewer</td>
  <td markdown="1">
ICML 2018, NeurIPS 2018, NeurIPS Deep RL Workshop 2018, ICLR 2019 (outstanding reviewer), ICML 2019, ICCV 2019
  </td>
</tr>
<tr>
  <td class='col-md-2'>Admissions</td>
  <td markdown="1">
CMU CSD MS 2014-2015
  </td>
</tr>
</table> -->


<!-- ## <i class="iconfont icon-laptop" style="font-size: 0.9em"></i> Skills
<table class="table table-hover">
<tr>
  <td class='col-md-2'>Languages</td>
  <td>
C, C++, Golang, Python, Make, <i>LaTeX</i>, Ruby, R, Javascript
  </td>
</tr>
<tr>
  <td class='col-md-2'>Frameworks</td>
  <td>
NumPy, Pandas, PyTorch, SciPy, TensorFlow
  </td>
</tr>
<tr>
  <td class='col-md-2'>Systems</td>
  <td>
Linux, OSX
  </td>
</tr>
</table>
 -->







<!-- for publication -->

## <i class="iconfont icon-Publications" style="font-size: 1.2em"></i> All Publications <a href="https://scholar.google.com/citations?user=842nSvkAAAAJ&hl=zh-CN" class="btn btn-primary" style="padding: 0.3em;"> <i class="iconfont icon-Googlescholar"></i> Google Scholar </a>

<h2>2021</h2>
<table class="table table-hover">

<tr style="background-color: #ffffd0">
<td>
    <strong>DevelSet: Deep Neural Level Set for Instant Mask optimization</strong><br>
    <strong>Guojin Chen</strong>, Ziyang Yu, Hongduo Liu, Yuzhe Ma, and Bei Yu<br>
    IEEE/ACM International Conference on Computer-Aided Design <strong>ICCAD 2021</strong><br>

    [C4]
<!-- [<a href='javascript:;' onclick='$("#abs_lambert2020objective_all_bib").toggle()'>abs</a>]
[<a href='data/papers/C1-ICCAD20-DAMO.pdf' target='_blank'>preprint</a>]
[<a href='https://arxiv.org/abs/2008.00806' target='_blank'>arXiv</a>]
[<a href='https://whova.com/portal/webapp/iccad_202011/Agenda/1273042/' target='_blank'>whova</a>]
[<a href="data/bibtex/damo_iccad_gjchen.bib" download="damo_iccad_gjchen.bib">bibtex</a>]
<br>
<div id="abs_lambert2020objective_all_bib" style="text-align: justify; display: none" markdown="1">
Continuous scaling of the VLSI system leaves a great challenge on manufacturing and optical proximity correction (OPC) is widely applied in conventional design flow for manufacturability optimization.
Traditional techniques conducted OPC by leveraging a lithography model and suffered from prohibitive computational overhead, and mostly focused on optimizing a single clip without addressing how to tackle the full chip.
In this paper, we present DAMO, a high performance and scalable deep learning-enabled OPC system for full chip scale.
It is an end-to-end mask optimization paradigm which contains a Deep Lithography Simulator (DLS) for lithography modeling and a Deep Mask Generator (DMG) for mask pattern generation.
Moreover, a novel layout splitting algorithm customized for DAMO is proposed to handle the full chip OPC problem.
Extensive experiments show that DAMO outperforms the state-of-the-art OPC solutions in both academia and industrial commercial toolkit.
</div> -->

</td>
</tr>


<tr>
<td>
    <strong>Learning Point Clouds in EDA.</strong> (Invited Paper) <br>
    Wei Li, <strong>Guojin Chen</strong>, Haoyu Yang, Ran Chen and Bei Yu<br>
    ACM International Symposium on Physical Design, <strong>ISPD 2021</strong><br>
    [C3]
[<a href='javascript:;' onclick='$("#abs_pceda_invited").toggle()'>abs</a>]
[<a href='http://www.cse.cuhk.edu.hk/~byu/papers/C116-ISPD2021-PointCloud.pdf' target='_blank'>preprint</a>]
[<a href='http://www.cse.cuhk.edu.hk/~byu/papers/C116-ISPD2021-PointCloud-slides.pdf' target='_blank'>slides</a>]
[<a href="data/bibtex/pceda_invited.bib" download="pceda_gjchen.bib">bibtex</a>]
<br>
<div id="abs_pceda_invited" style="text-align: justify; display: none" markdown="1">
The exploding of deep learning techniques have motivated the development in various fields, including intelligent EDA algorithms from physical implementation to design for manufacturability. Point cloud, defined as the set of data points in space, is one of the most important data representations in deep learning since it directly preserves the original geometric information without any discretization.
However, there are still some challenges that stifle the applications of point clouds in the EDA field.
In this paper, we first review previous works about deep learning in EDA and point clouds in other fields.
Then, we discuss some challenges of point clouds in EDA raised by some intrinsic characteristics of point clouds.
Finally, to stimulate future research, we present several possible applications of point clouds in EDA and demonstrate the feasibility by two case studies.
</div>
</td>
</tr>
</table>


<h2>2020</h2>
<table class="table table-hover">

<tr>
<td>
    <strong>A GPU-enabled Level Set Method for Mask Optimization</strong><br>
    Ziyang Yu, <strong>Guojin Chen</strong>, Yuzhe Ma and Bei Yu<br>
    IEEE/ACM Proceedings Design, Automation and Test in Europe, <strong>DATE 2021</strong><br>
    [C2]
[<a href='javascript:;' onclick='$("#abs_levelset_all_bib").toggle()'>abs</a>]
[<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C115-DATE2021-LevelSet.pdf' target='_blank'>preprint</a>]
[<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C115-DATE2021-LevelSet-slides.pdf' target='_blank'>slides</a>]
<br>
<div id="abs_levelset_all_bib" style="text-align: justify; display: none" markdown="1">
As the feature size of advanced integrated circuits keeps shrinking, resolution enhancement technique (RET) is utilized to improve the printability in the lithography process.
Optical proximity correction (OPC) is one of the most widely used RETs aiming at compensating the mask to generate a more precise wafer image.
In this paper, we put forward a level-set based OPC with high mask optimization quality and fast convergence.
In order to suppress the disturbance of the condition fluctuation in lithography process, we propose a new process window-aware cost function.
Then, a novel momentum-based evolution technique is adopted, which demonstrates substantial improvement.
Moreover, graphics processing unit (GPU) is leveraged for accelerating the proposed algorithm.
Experimental results on ICCAD 2013 benchmarks show that our algorithm outperforms all previous OPC algorithms in terms of both solution quality and runtime overhead.
</div>

</td>
</tr>


<tr style="background-color: #ffffd0">
<td>
    <strong>DAMO: Deep Agile Mask Optimization for Full Chip Scale</strong><br>
    <strong>Guojin Chen</strong>, Wanli Chen, Yuzhe Ma, Haoyu Yang and Bei Yu<br>
    IEEE/ACM International Conference on Computer-Aided Design <strong>ICCAD 2020</strong><br>

    [C1]
[<a href='javascript:;' onclick='$("#abs_lambert2020objective_all_bib").toggle()'>abs</a>]
[<a href='data/papers/C1-ICCAD20-DAMO.pdf' target='_blank'>preprint</a>]
[<a href='https://arxiv.org/abs/2008.00806' target='_blank'>arXiv</a>]
[<a href='https://whova.com/portal/webapp/iccad_202011/Agenda/1273042/' target='_blank'>whova</a>]
[<a href="data/bibtex/damo_iccad_gjchen.bib" download="damo_iccad_gjchen.bib">bibtex</a>]
<br>
<div id="abs_lambert2020objective_all_bib" style="text-align: justify; display: none" markdown="1">
Continuous scaling of the VLSI system leaves a great challenge on manufacturing and optical proximity correction (OPC) is widely applied in conventional design flow for manufacturability optimization.
Traditional techniques conducted OPC by leveraging a lithography model and suffered from prohibitive computational overhead, and mostly focused on optimizing a single clip without addressing how to tackle the full chip.
In this paper, we present DAMO, a high performance and scalable deep learning-enabled OPC system for full chip scale.
It is an end-to-end mask optimization paradigm which contains a Deep Lithography Simulator (DLS) for lithography modeling and a Deep Mask Generator (DMG) for mask pattern generation.
Moreover, a novel layout splitting algorithm customized for DAMO is proposed to handle the full chip OPC problem.
Extensive experiments show that DAMO outperforms the state-of-the-art OPC solutions in both academia and industrial commercial toolkit.
</div>

</td>
</tr>



</table>


## <i class="iconfont icon-presentation" style="font-size: 0.9em"></i> Presentation & Talks
<table class="table table-hover">
<tr>
  <td class='col-md-2'>Mar 2020</td>
  <td>
    <a href="data/slides/20200321-fft.pdf" target="_blank">CUDA based Convolution and FFT on OPC</a>
  </td>
</tr>
<tr>
  <td class='col-md-2'>May 2020</td>
  <td>
    <a href="data/slides/20200514-opc.pdf" target="_blank">DLS-DMO: High Accuracy DL-Based OPC With DLS</a>
  </td>
</tr>
<!-- <tr>
  <td class='col-md-2'>Jun - 2020</td>
  <td>
    <a href="data/slides/pchsd-20200622.pdf" target="_blank">Point Cloud Hotspot Detection</a>
  </td>
</tr> -->
</table>

## <i class="iconfont icon-award-solid" style="font-size: 0.9em"></i> Honors & Awards
<table class="table table-hover">
<tr>
  <td class='col-md-2'>Apr 2021</td>
  <td>
    Undergraduate Academic Competition Merit Scholarship, BJTU.
  </td>
</tr>
<tr>
  <td class='col-md-2'>Oct 2020</td>
  <td>
    Third Prize of Scholarship on Social Works, BJTU.
  </td>
</tr>
<tr>
  <td class='col-md-2'>Oct 2020</td>
  <td>
    Second Grade Academic Excellence Scholarship, BJTU.
  </td>
</tr>
<tr>
  <td class='col-md-2'>Sep 2020</td>
  <td>
    International College Students'- “Internet +” Innovation and Entrepreneurship Competition in Beijing municipal-level.
  </td>
</tr>
<tr>
  <td class='col-md-2'>Nov 2019</td>
  <td>
    Special Prize of The 14th “Dongfeng Nissan Cup”- National Industrial Engineering Application Case Competition, with rank of 1/320.
  </td>
</tr>
</table>



## <i class="iconfont icon-CertificateInformat" style="font-size: 0.9em"></i> Certificates
<table class="table table-hover">
<tr>
  <td class='col-md-2'>Nov 2020</td>
  <td>
    <a href="data/certificates/2020_ICCAD_CAD_speaker_certificate.pdf" target="_blank">ICCAD 2020 Speaker Certificate of Appreciation Presented to <strong>Guojin Chen</strong>.</a>
  </td>
</tr>
<!-- <tr>
  <td class='col-md-2'>Jun - 2020</td>
  <td>
    <a href="data/slides/pchsd-20200622.pdf" target="_blank">Point Cloud Hotspot Detection</a>
  </td>
</tr> -->
</table>

<!--
<h2>2014</h2>
<table class="table table-hover">

<tr>
<td>
    <strong>Performance study of Spindle, a web analytics query engine implemented in Spark</strong><br>
    <strong>B. Amos</strong> and D. Tompkins<br>
    CloudCom 2014<br>

    [1]
[<a href='javascript:;'
    onclick='$("#abs_amos2014performance_all_bib").toggle()'>abs</a>] [<a href='http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7037709' target='_blank'>pdf</a>]  [<a href='https://github.com/adobe-research/spindle' target='_blank'>code</a>] <br>

<div id="abs_amos2014performance_all_bib" style="text-align: justify; display: none" markdown="1">
This paper shares our experiences building and benchmarking Spindle as an open
source Spark-based web analytics platform. Spindle's design has been
motivated by real-world queries and data requiring concurrent, low latency
query execution. We identify a search space of Spark tuning options and study
their impact on Spark's performance. Results from a self-hosted six node
cluster with one week of analytics data (13.1GB) indicate tuning options such
as proper partitioning can cause a 5x performance improvement.
</div>

</td>
</tr>


<tr>
<td>
    <strong>Global Parameter Estimation for a Eukaryotic Cell Cycle Model in Systems Biology</strong><br>
    T. Andrew, <strong>B. Amos</strong>, D. Easterling, C. Oguz, W. Baumann, J. Tyson, and L. Watson<br>
    SummerSim 2014<br>

    [2]
[<a href='javascript:;'
    onclick='$("#abs_andrew2014global_all_bib").toggle()'>abs</a>] [<a href='http://dl.acm.org/citation.cfm?id=2685662' target='_blank'>pdf</a>] <br>

<div id="abs_andrew2014global_all_bib" style="text-align: justify; display: none" markdown="1">
The complicated process by which a yeast cell divides, known as the cell
cycle, has been modeled by a system of 26 nonlinear ordinary differential
equations (ODEs) with 149 parameters. This model captures the chemical
kinetics of the regulatory networks controlling the cell division process
in budding yeast cells. Empirical data is discrete and matched against
discrete inferences (e.g., whether a particular mutant cell lives or dies)
computed from the ODE solution trajectories. The problem of
estimating the ODE parameters to best fit the model to the data is a
149-dimensional global optimization problem attacked by the deterministic
algorithm VTDIRECT95 and by the nondeterministic algorithms differential
evolution, QNSTOP, and simulated annealing, whose performances are
compared.
</div>

</td>
</tr>


<tr>
<td>
    <strong>Fortran 95 implementation of QNSTOP for global and stochastic optimization</strong><br>
    <strong>B. Amos</strong>, D. Easterling, L. Watson, B. Castle, M. Trosset, and W. Thacker<br>
    SpringSim (HPC) 2014<br>

    [3]
[<a href='javascript:;'
    onclick='$("#abs_amos2014fortran_all_bib").toggle()'>abs</a>] [<a href='http://dl.acm.org/citation.cfm?id=2663525' target='_blank'>pdf</a>] <br>

<div id="abs_amos2014fortran_all_bib" style="text-align: justify; display: none" markdown="1">
A serial Fortran 95 implementation of the QNSTOP algorithm is presented.
QNSTOP is a class of quasi-Newton methods for stochastic optimization with
variations for deterministic global optimization. This discussion provides
results from testing on various deterministic and stochastic optimization
functions.
</div>

</td>
</tr>


<tr>
<td>
    <strong>QNSTOP: Quasi-Newton Algorithm for Stochastic Optimization</strong><br>
    <strong>B. Amos</strong>, D. Easterling, L. Watson, W. Thacker, B. Castle, and M. Trosset<br>
    ACM TOMS 2014<br>

    [4]
[<a href='javascript:;'
    onclick='$("#abs_amos2014QNSTOP_all_bib").toggle()'>abs</a>] [<a href='https://dl.acm.org/doi/10.1145/3374219' target='_blank'>pdf</a>] <br>

<div id="abs_amos2014QNSTOP_all_bib" style="text-align: justify; display: none" markdown="1">
QNSTOP consists of serial and parallel (OpenMP) Fortran 2003 codes for the
quasi-Newton stochastic optimization method of Castle and Trosset. For
stochastic problems, convergence theory exists for the particular
algorithmic choices and parameter values used in QNSTOP. Both the parallel
driver subroutine, which offers several parallel decomposition strategies, and the serial driver subroutine can be used for stochastic optimization or
deterministic global optimization, based on an input switch. QNSTOP is
particularly effective for “noisy” deterministic problems, using only
objective function values. Some performance data for computational systems
biology problems is given.
</div>

</td>
</tr>

</table> -->

<!--
<h2>2013</h2>
<table class="table table-hover">

<tr>
<td>
    <strong>Applying machine learning classifiers to dynamic Android malware detection at scale</strong><br>
    <strong>B. Amos</strong>, H. Turner, and J. White<br>
    IWCMC 2013<br>

    [1]
[<a href='javascript:;'
    onclick='$("#abs_amos2013applying_all_bib").toggle()'>abs</a>] [<a href='http://bamos.github.io/data/papers/amos-iwcmc2013.pdf' target='_blank'>pdf</a>]  [<a href='https://github.com/VT-Magnum-Research/antimalware' target='_blank'>code</a>] <br>

<div id="abs_amos2013applying_all_bib" style="text-align: justify; display: none" markdown="1">
The widespread adoption and contextually sensitive
nature of smartphone devices has increased concerns over smartphone
malware. Machine learning classifiers are a current method
for detecting malicious applications on smartphone systems. This
paper presents the evaluation of a number of existing classifiers, using a dataset containing thousands of real (i.e. not synthetic)
applications. We also present our STREAM framework, which
was developed to enable rapid large-scale validation of mobile
malware machine learning classifiers.
</div>

</td>
</tr>

</table> -->
