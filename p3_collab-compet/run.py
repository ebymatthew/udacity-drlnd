from unityagents import UnityEnvironment

import time
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent, ReplayBuffer
import logging

import argparse

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='1.0.0')
parser.add_argument('--episodes', default=30, help='number of episodes', type=int)
parser.add_argument('--env', default='./Tennis.app', help='Path to the Tennis Unity environment')


def run(
        env_location,
        n_episodes=30,
    ):

    env = UnityEnvironment(file_name=env_location)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    logger.info(f'Number of agents: {num_agents}')

    # size of each action
    action_size = brain.vector_action_space_size
    logger.info(f'Size of each action: {action_size}')

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    logger.info('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    logger.info(f'The state for the first agent looks like: {states[0]}')

    # reset the environment

    # Replay memory
    random_seed = 2
    memory0 = ReplayBuffer(action_size, 128, 128, random_seed)

    def create_agent(memory):
        return Agent(
            state_size=states.shape[1],
            action_size=brain.vector_action_space_size,
            random_seed=random_seed,
            memory=memory,
            batch_size=128
        )

    agent0 = create_agent(memory0)
    agent1 = create_agent(memory0)

    # load saved weights
    agent0.actor_local.load_state_dict(torch.load('checkpoint_actor0.pth', map_location='cpu'))
    agent1.actor_local.load_state_dict(torch.load('checkpoint_actor1.pth', map_location='cpu'))

    def ddpg(n_episodes, average_window=100):
        scores_deque = deque(maxlen=average_window)
        scores_all = []
        average_scores_all = []

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=False)[brain_name]
            states = np.array(env_info.vector_observations, copy=True)                  # get the current state (for each agent)
            agent0.reset()
            agent1.reset()
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)

            while True:
                action0 = agent0.act(states[0], add_noise=False)
                action1 = agent1.act(states[1], add_noise=False)
                actions = np.concatenate((action0, action1))

                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                dones = env_info.local_done                        # see if episode finished

                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                any_done = np.any(dones)
                assert any_done == np.all(dones)
                if any_done:                                  # exit loop if episode finished
                    break

            score_episode = np.max(scores)
            best_agent = np.argmax(scores)
            scores_deque.append(score_episode)
            scores_all.append(score_episode)
            average_score_queue = np.mean(scores_deque)
            average_scores_all.append(average_score_queue)

            logger.info('\rEpisode {}\tScore: {:.4f}\tBest Agent: {}\tAverage Score: {:.4f}'.format(i_episode, score_episode, best_agent, average_score_queue))
            if i_episode > average_window and average_score_queue > 1.0:
                break

        return scores_all, average_scores_all

    scores, average_scores = ddpg(n_episodes=n_episodes)

    env.close()

    return np.max(average_scores)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.env, n_episodes=args.episodes)
    logger.info(f'Done - elapsed time: {time.process_time()} seconds')
