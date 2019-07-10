from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import itertools
import time
import torch
import torch.cuda
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from concurrent import futures

from ddpg_agent import Agent, ReplayBuffer
import logging

import argparse

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='1.0.0')
parser.add_argument('--episodes', default=300, help='number of episodes', type=int)
parser.add_argument('--env', default='./Reacher_Linux_NoVis20/Reacher.x86', help='Path to the Reacher Unity environment')
parser.add_argument('--curve', default='learning.curve.png', help='Location to output learning curve')


def grouper(n, iterable):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    grouped_padded = itertools.zip_longest(fillvalue=None, *args)
    grouped_filtered = (x for x in grouped_padded if x is not None)
    return grouped_filtered


def train(env_location, curve_path, n_episodes=1000):
    env = UnityEnvironment(file_name=env_location)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

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
    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 1024        # minibatch size
    random_seed = 2
    memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def create_agent(i):
        return Agent(state_size=states.shape[1], action_size=brain.vector_action_space_size, random_seed=random_seed, memory=memory, batch_size=BATCH_SIZE, index=i)

    agents = [create_agent(i) for i in range(20)]

    def ddpg(n_episodes, max_t=300, print_every=100):
        scores_deque = deque(maxlen=print_every)
        scores_all = []

        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        ex = futures.ThreadPoolExecutor(max_workers=device_count)

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = np.array(env_info.vector_observations, copy=True)                  # get the current state (for each agent)
            # state = env.reset()
            for agent in agents:
                agent.reset()
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)

            # state = states[0]

            for t in range(max_t):

                actions = list()
                for device_batch in grouper(device_count, range(num_agents)):
                    batch_actions = ex.map(lambda idx: agents[idx].act(states[idx]), device_batch)
                    actions.extend(batch_actions)
                    #actions = np.array([agents[i].act(states[i]) for i, state in enumerate(states)])
                actions = np.array(actions)
                # print(f'action: {action}')

                # actions = np.array([action]) # temporarily rename
                # actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1

                #print(f'actions: {actions}')
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                #print(f'dones: {dones}')
                #scores += env_info.rewards                         # update the score (for each agent)
                #print(f'scores: {scores}')

                # Add experience to replay buffer for all agents

                def add_to_memory(idx):
                    reward = rewards[idx] # temporarily rename
                    next_state = next_states[idx] # temporarily rename
                    done = dones[idx] # temporarily rename
                    action = actions[idx]
                    memory.add(states[idx], action, reward, next_state, done)

                for device_batch in grouper(device_count, range(num_agents)):
                    batch_actions = ex.map(lambda idx: add_to_memory(idx), device_batch)
                    list(batch_actions)  # convert to list to wait

                '''
                for i in range(num_agents):
                    reward = rewards[i] # temporarily rename
                    next_state = next_states[i] # temporarily rename
                    done = dones[i] # temporarily rename
                    action = actions[i]
                    memory.add(states[i], action, reward, next_state, done)

                    #print(f'state: {states[i]}')
                    #print(f'action: {action}')
                    #print(f'reward: {reward}')
                    #print(f'next_state: {next_state}')
                    #print(f'done: {done}')
                '''

                for device_batch in grouper(device_count, range(num_agents)):
                    batch_actions = ex.map(lambda idx: agents[idx].step(), device_batch)
                    list(batch_actions)  # convert to list to wait

                # for i in range(num_agents):
                #    agents[i].step()

                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                assert np.any(dones) == np.all(dones), "Not all done at the same time"
                if np.any(dones):                                  # exit loop if episode finished
                    break
            scores_deque.append(scores)
            scores_all.append(np.mean(scores))
            # print(f'Dequeue: {scores_deque} mean: {np.mean(scores_deque)}')

            average_score = np.mean(scores_deque)

            logger.info('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            if average_score > 30:
                break
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor2.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic2.pth')
            if i_episode % print_every == 0:
                logger.info('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        return scores_all

    scores = ddpg(n_episodes=n_episodes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('learning.curve.png')

    env.close()


if __name__ == '__main__':
    args = parser.parse_args()
    train(args.env, args.curve, n_episodes=args.episodes)
    logger.info(f'Done - elapsed time: {time.process_time()} seconds')
