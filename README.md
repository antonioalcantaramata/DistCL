# DistCL: A Neural Network-Based Distributional Constraint Learning tool for Mixed-Integer Stochastic Optimization

DistCL extent the constraint learning methodology in stochastic mixed-integer optimization by addressing the statistical uncertainty in the response variables. 

DistCL helps practitioners in the following steps:

1. Training a neural network-based model to estimate the parameters of the conditional distribution of a given variable $Y$ dependent on own decisions $X$ and contextual information $\theta$
2. Tranform the structure of the neural network into a piece-wise linear set of constraints.
3. Embed this constraint within a Mixed-Integer Stochastic Optimization problem and generate scenarios in a linear way.
