# Deep Q-Learning (DQN)

(Possibly) infinite/ continuous state space but discrete action space.

## Prediction problem

We are given a policy.
- Target: estimation of the return (r + gamma * V(s')).
- Prediction: V(s) = W^T*s + b.
- Parameters to update: weight matrix (W) and bias vector (b) (i.e. the linear regression parameters - use squared error and do gradient descent to update these parameters).

## Control problem

Q is a model prediction (model.predict()).
- Target: y = r + gamma * max(a')Q(s',a')  (read: "max over all actions a' of Q(s',a')").
- Prediction: Q(s,a).
- Parameters to update: W, b.

## Deep Q-Learning

- Using gradient descent for all parameters (all weights and biases) in the model (done with automatic differentiation for simplicity).
- Experience replay buffer/ memory (randomised) (which hold "transitions").
- Now using batch gradient descent (looking at multiple samples at once) instead of stochastic gradient descent (looking at one sample at a time).



### Notes
- For continuous/ infinite state or action spaces, we use machine learning (function approximation) instead of tabular methods.
