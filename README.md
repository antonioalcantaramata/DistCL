# DistCL: A Neural Network-Based Distributional Constraint Learning tool for Mixed-Integer Stochastic Optimization

DistCL extends the constraint learning methodology in stochastic mixed-integer optimization by addressing the statistical uncertainty in the response variables. 

DistCL helps practitioners in the following steps:

1. Training a neural network-based model to estimate the parameters of the conditional distribution of a given variable $Y$ dependent on own decisions $X$ and contextual information $\theta$
2. Tranform the structure of the neural network into a piece-wise linear set of constraints.
3. Embed these constraints within a Mixed-Integer Stochastic Optimization problem and generate scenarios in a linear way.


See the paper [A Neural Network-Based Distributional Constraint Learning Methodology for Mixed-Integer Stochastic Optimization](https://arxiv.org/abs/2211.11392) for more information about the developed methodology and the case study for a real-world example. If you use this software or the methodology, you can cite it as:

> Alcántara, A., & Ruiz, C. (2022). A Neural Network-Based Distributional Constraint Learning Methodology for Mixed-Integer Stochastic Optimization. arXiv preprint arXiv:2211.11392.


