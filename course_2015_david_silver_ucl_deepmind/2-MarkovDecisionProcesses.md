# Markov Decision Processes

State Transition Probability Matrix (Pss'): probability of transitioning into all successor states s', starting from a given state s.
    |P11 ... P1n| 
    |...     ...|
    |Pn1 ... Pnn|

Markov Process/Chain: a sequence of states with Markov property (sampled from matrix) (S)
    Due to Markov property, no need to store all history, only current state
    Chains of random sequence, which are Markov
    (Any sample by definition is finite, even if the decision process has infinite loops; they all terminate)

(Optimal control: continuous MDPs
Partially observable environments can be converted to MDPs)

## Markov Reward Process (MRP)

Accumulated reward from sequence sampled.

Reward function (R): how much immediate reward we get from a given state at t+1. 
    To then do maximum accumulated sum of these rewards

Discount factor (y): Present value of future rewards. 
    Between 0 and 1, with 0 meaning shotsightedness/myopia (only next step considered) and 1 far-sighted/undiscounted. 
    Prevents G from being infinite.
    Benefits of having one:
        Mathematically convenient. Increasing uncertainty the further we go into the future. Imperfection of model. Avoids infinite returns in cyclic Markov processes. (Could be considered as the reverse of an interest rate). (Humans and animals exhibit hyperbolic, not exponential, discounting).

Return/Goal (Gt): total discounted reward from time step t, for a sample sequence of MRP
    Gt = Rt+1 + y * Rt+1 + ... = Summation(from 0 til inf) y^k * Rt+k+1
    (G is random, so there is no expectation)

## Value Function

How good it is to be in a certain state long-term. Expected return starting from state s.
    v(s) = E[Gt | St=s]
    (No maximisation, just calculating it, sometimes by averaging the G from different sample sequences, and taking probabilities into account)

## Bellman Equation

The value function can be decomposed into the immediate expected reward (Rt+1) and the value of where you end up (i.e. the discounted value of successor state: y * v(st+1))

v(s) = E[Rt+1 + y * v(st+1) | St=s]

    (Sometimes Rt is used in literature instead of Rt+1)
    1-step look-ahead tree and then averaging over all things that might happen next
    (Related to the law of iterative expectations)

Using a vector/matrix representation, and dot product:

V = R + y * P * V

where V is the vector containing the value of all states, R the reward from all states, and P the state transition probability matrix

This can be expressed as a linear equation for evaluation, not for maximisation: 

    R = (1 - y * P) * V
    v = (1 - y * P)^-1 * R

    However, matrix inversion is not practical for large MRPs due to computational complexity (O(n^3)). In such cases, we
    use dynamic programming, Monte-Carlo evaluation, temporal-difference learning

## Markov Decision Process (MDP)

S is finite, but can be made continuous
A: finite set of actions
P (the state transition probability matrix) now depends on actions too
    Pss'^a = P[St+1=s' | St=s, At=a], one for each action
Rs^a can depend on actions as well

We wanna maximise the sum of rewards

Policy (phi): distribution over actions given a state

    phi(a|s) = P[At=a | St=s]

    (Stationary, i.e. time-independent each time step, or stochastic)
    (Future rewards aren't required explicitly due to its markovianness)
    (A stochastic transition matrix is good for exploration)

    (An MRP or MP can always be recovered by flattening an MDP into a Markov chain, given our current policy.
    The sequence of states is a Markov process/chain, regardless of policy chosen.
    Transition dynamics and reward function == averaging over a policy.
    One linear equation, so solvable, but not very efficient).

State Value Function: expected return/total reward, how good it is to be in state s following policy phi, from that point onwards

    vphi(s) = Ephi[Gt | St=s]

Action Value Function: all of above, plus also dependent on following action a

    qphi(s,a) = E[Gt | St=s, At=a]

    Averaging possible actions using probabilities of transition dynamics

Bellman Expectation Equation
    The value function can be decomposed into the immediate reward and the discounted value of successor state (vphi(s')), following a policy (vphi(s)) and taking an action (qphi(s,a))
    (The probabilities are defined by the policy)

    qphi(s,a) = Rs^a + y * Summation(s' in S) Pss'^a * vphi(s')

    (Stitched together using recursion)

    vphi(s) = Summation(a in A) phi(a|s) * (Rs^a + y * Summation(s' in S) Pss'^a * vphi(s'))
    (Eg. 2-step look-ahead, state-actions-states or action-state-actions)

## Optimal Behaviour

Finding the best way to behave, maximum possible reward we can extract

Optimal Value Function
    
    v*(s) = max(phi) vphi(s), maximum value function over all policies

Optimal Action Value Function
    q*(s,a), given an action from a state, compared to reach optimal behaviour within MDP

    chosen a = argmarx(a in A) q*(s,a)

Optimal Policy
    (Stochastic mapping from states to actions)
    Its vphi(s) must be the best in all states
    There is always an optimal policy for any MDP, but it's not necessarily unique, there can be more than one

    phi*(a,s) such that chosen a
    (Deterministic optimal policy)

(Working backwards is good for Bellman Optimality Equation, finding optimal value functions recursively)

Getting q from v (starting with a state):
    One-step look-ahead, max q, not average:
    v*(s) = max(a) q*(s,a)

    Two-step look-ahead: max of actions, then average of v*(s')
    v*(s) = max(a) Rs^a + y * Summation(s' in S) Pss'^a * v*(s')
    (Principle of optimality: optimal behaviour can be achieved by behaving optimally in next step and then for the rest of the trajectory)

Getting v from q (starting with an action):
    (Using average. Considers environment dynamics)
    q*(s,a) = Rs^a + y * Summation(s' in S) Pss'^a * v*(s')

    Bellman Optimality Equation for q*
        q*(s,a) = Rs^a + y * Summation(s' in S) Pss'^a * max(a') q*(s',a')
        2-step look-ahead: average, then max q*(s',a')
        It isn't linear (requires matrix inversion), so it's often solved using iterative solution methods, such as dynamic programming, value iteration, policy iteration, Q-learning, Sarsa, to take it from n^3 to n^2

Extras:
How to deal with uncertainty in MDP model?
    Either explicitly represent it, Baysian, solve for a distribution of MDPs (computationally expensive), or add uncertainty in MDP. Discount factors can be variable as well
Adding risk:
    A cost on variance can be added for risk-sensirive MDPs

- Infinite MDPs
- Continuous MDPs
- POMDPs (partially-observable MDPs)
- Undiscounted MDPs
