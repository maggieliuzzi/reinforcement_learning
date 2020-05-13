# Q-Learning

Control algorithm. We choose an action (based on Q and s) and then we update Q(s,a).

Notes:
- When updating Q(s,a) we assume we will take the greedy action (off-policy, the update may not match the actions taken).
- Doing (Partial) gradient descent on the Q-table allows us to update Q at every step, instead of having to wait til the end of an episode (if there is one) to do so. We only need to wait to receive a single reward to make an update. "Online learning", the agent learns as data is collected.

## Concepts

- Temporal Diference: while Monte Carlo was an approximation to the expected value problem, Temporal Difference is an approximation to Monte Carlo.
The return can be defined recursively, so we approximate it with the next reward and the value at the next state.

- Epsilon-Greedy Approach

## Subsections

- Deep Q-Learning
