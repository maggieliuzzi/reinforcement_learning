# The base of this script was sourced from "Advanced AI: DRL with Python" by LazyProgrammer
'''
Random Search:
Choosing random weight vectors.
'''

from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers


NUM_WEIGHTS = 10
NUM_EPISODES = 10


def get_action(s, w):
	return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params): 
	observation = env.reset()
	done = False
	t = 0

	while not done and t < 10000:  # new versions of gym auto stop
		# env.render()
		t += 1
		action = get_action(observation, params)
		observation, reward, done, info = env.step(action)
		if done:
			break

	return t


def play_multiple_episodes(env, T, params):  # testing potential new weights T times before updating the official ones
	episode_lengths = np.empty(T)  # T: 100, number of episodes we test new weights for

	for i in range(T):
		episode_lengths[i] = play_one_episode(env, params)

	avg_length = episode_lengths.mean()  # mean of 
	print("avg length:", avg_length)

	return avg_length


def random_search(env, num_weights, num_episodes):
	episode_lengths = []
	best = 0
	params = None
	for t in range(num_weights):
		new_params = np.random.random(4)*2 - 1  # choosing random weight vectors
		avg_length = play_multiple_episodes(env, num_episodes, new_params)
		episode_lengths.append(avg_length)

		if avg_length > best:
			params = new_params
			best = avg_length
			
	return episode_lengths, params



if __name__ == '__main__':

	env = gym.make('CartPole-v0')
	env = wrappers.Monitor(env, 'videos')

	num_weights = NUM_WEIGHTS
	num_episodes = NUM_EPISODES

	episode_lengths, params = random_search(env, num_weights, num_episodes)
	plt.title("Avg game length for each set of weights tried")
	plt.plot(episode_lengths)
	plt.show()

	# play a final set of episodes
	print("***Final run with final weights***")
	play_multiple_episodes(env, num_episodes, params)
