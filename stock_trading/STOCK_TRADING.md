# Stock Trading using Deep Learning


Paper: "Practical Deep Learning Approach for Stock Trading" (2018, https://arxiv.org/pdf/1811.07522.pdf)
Instead of using DDPG, we use DQN.

## State
A vector (length: 2N+1, with N=3) (normalised, i.e. zero mean and unit variance) combining:
- How many shares of each stock I own (N stocks)
- Current price of each stock (N stocks)
- How much uninvested cash we have (1 value)

## Actions
N^3 possible actions (27 in this case, since N=3). One action will mean performing all steps in the specified order.
- Sell
- Buy
- Hold

### Assumptions/ simplifications:
- No transaction costs
- If we sell a stock, we sell all the shares we own for a particular stock
- If we buy one stock we buy as many shares as possible
- If we buy multiple stocks, we loop through every stock and buy one share of each until we run out of money (round robin, to avoid Knapsack problem)
- Selling before buying

## Reward
- Change in the value of our portfolio from state s to state s', where the value = summation(num of stocks owned * stock price) + cash = s^T*p + c, where s and p are vectors

## Agent
- Replay buffer to store its transitions in the environment
- Model (eg. neural network) to approximate Q values
- Given state decide action to perform (calculates Q(s,a) and takes the argmax over all possible actions)
- Gets a random sample from the replay buffer and uses it to calculate a supervised learning dataset which consists of input and target pairs to then train our model by running one iteration of gradient descent over a batch of data
- During training, the agent stores states, actions and rewards and perform Q-Learning updates in order to train the Q function approximator

## Replay Buffer
During training, we sample "transitions" from the replay buffer randomly, calculate input-target pairs and do one step of gradient descent.
To avoid memory leaks with bigger datasets, we create our own circular buffer, using fixed-size arrays (so that space complexity is constant) and using a pointer to keep track of the oldest value.
