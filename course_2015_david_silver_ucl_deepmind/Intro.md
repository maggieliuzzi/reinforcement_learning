# Introduction to Reinforcement Learning

Sequential Decision Making

Given observations (Ot) and rewards (Rt, number), choose actions (At)

History (Ht): time series, experience
State (St): a function of the history (St = f(Ht))
- Environment state: not seeing by agent
- Agent state: subjective representation of the environment and its state

Markovianness of State: the current state contains all relevant information from the past; the future (including prediction of future rewards) is independent of the past given the present; the state is a sufficient statistic of the future.
    P[St+1 | St] = P[St+1 | S1...St]
    (The environment state, St^e, is always Markov)

Observability of Environment State:
- Fully observable: Ot = St^a = St^e, Markov Decision Process (MDP)
- Partially observable: Partially-observable MDP (POMDP) (eg. robot with a camera which doesn't know its absolute location in the environment)

Agent's state representation (St^a):
Eg.
- Complete history: St^a = Ht
- Beliefs of environment state (eg. using full probability distribution, vector of probabilities)
- Recurrent neural network: linear combination of previous agent state and latest observations, plus some non-linearity around the whole thing


## RL Agents

### Components 
(Not all are required)

- Policy
    Map from state to action
    - Deterministic (a = f(s))
    - Stochastic: probability of taking a particular action given being in a specific state (f(a | s) = P[A=a | S=s]) (learning from experience so as to maximise reward) (useful for random exploration decisions to see more of the state space)

- Value Function
    Prediction of future reward if we follow this behaviour, useful for comparison (evaluation of the goodness of states)
    Vf(s) = Ef[Rt + y*Rt+1 + y^2*Rt+2 + ... | St=s]. Discounting factor to give more prominence to recent rewards (no need for a discrete horizon) (risk is implicitly taken into account but is sometimes added explicitly for some trading applications)
    - State Value Function (V)
    - Action Value Function (Q)

- Model
    (Optional: many model-free)
    Prediction of future behaviour of environment
    - Transitions: prediction of next state (i.e. dynamics, eg. position, angle of a helicopter)
        State Transition Model: probability of being in next state given current/previous state and action (Pss'^a = P[S'=s' | S=s, A=a])
    - Rewards: prediction of next, immediate reward
        Reward Model: expected reward given current state and action (Rs^a = E[R | S=s, A=a]

Eg. Maze:
    Rewards: -1 per time step (to make it do it as quickly as possible), Actions: N, E, S, W, States: agent's location in grid
    Policy: f(s) for each state (eg. if in this state, go left, if in this one, go right) (mapped using a data structure) (represented with arrows in diagram)
    Value Function: helps compare actions (eg. at intersections)
    Model: short-term, one-step prediction, immediate reward from each state is -1 (in action-dependent rewards, each action is given a different reward)

## Categorisation of RL agents
- Value Based: stores value function (policy is implicit) + model
- Policy Based: stores policy (no value function) + model
- Actor Critic: stores value function and policy + model
- Model Free: policy and/or value function (no model) (learns from experience)
- Model Based: policy and/or value function + model (it is given a model of the dynamics of the environment, eg. the dynamics of the helicopter)


## Key Problems
- Learning vs Planning
    How to improve policy to maximise future reward.
    Learning:
        Trial and error
        Environment initially unknown, a lot of interaction with the environment
    Planning:
        Known (perfect) model of the environment (eg. it is given differential equations describing wind)
        Internal computations (often using dynamic programming, possibly doing look-ahead/tree search), delayed external interaction
        (Can be used as a first step for Learning as well)
- Exploration and Exploitation
    Trade-off, so a balance is required
    Exploitation:
        Already discovered information, possibly suboptimal reward (eg. advertiser showing their most successful ad, drilling in a known location, going to your favourite restaurant, playing a known good move in a game)
    Exploration:
        Often losing immediate reward in order to get more environment information
- Prediction and Control
    Prediction: evaluating future given a policy (eg. the current one)
    Control: optimising future, finding the best policy
    (Prediction helps evaluate all policies to then choose the best one)


## Differences between RL and other Machine Learning paradigms
- No supervisor, only a reward signal
- Delayed, non-instantaneous feedback
- Time really matters (sequential, non-i.d.d data)
- Agent's actions affect the subsequent data it receives
