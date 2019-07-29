from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import functools
import time
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os.path

from ddpg_agent import Agent, ReplayBuffer
import logging

import argparse
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from train import train

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='1.0.0')
#parser.add_argument('--episodes', default=300, help='number of episodes', type=int)
parser.add_argument('--env', default='./Reacher_Linux_NoVis20/Reacher.x86', help='Path to the Reacher Unity environment')
parser.add_argument('--curve', default='./', help='Directory location to output learning curve')
parser.add_argument('--logs', default="./logs.json", help='Directory location to output learning curve')


def plot_curve(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('learning.curve.png')


def parameter_search(env_location, curve_path, logs):
    # Bounded region of parameter space
    pbounds = {
        # 'batch_size': (64, 4096),
        # 'buffer_size': (int(1e4), int(1e8)),
        'gamma': (0.98, 0.9999),
        'tau': (5e-4, 5e-3),
        'lr_actor': (5e-5, 5e-3),
        'lr_critic': (5e-5, 5e-4),
        # 'weight_decay': (0.00001, 0.001),
        'update_every': (3, 15),
        'num_updates': (1, 5),
    }

    # apply static params to train function
    def train_func(
            batch_size=2048,
            buffer_size=int(1e6),
            gamma=0.99,  # discount factor
            tau=1e-3,  # for soft update of target parameters
            lr_actor=1e-4,  # learning rate of the actor
            lr_critic=1e-4,  # learning rate of the critic
            weight_decay=0.0001,  # L2 weight decay
            update_every=10,  # how often to update the network
            num_updates=2  # how many updates to perform
    ):
        return train(
            env_location=env_location,
            curve_path=curve_path,
            n_episodes=1000,
            buffer_size=buffer_size,
            batch_size=int(batch_size),
            gamma=gamma,
            tau=tau,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            weight_decay=weight_decay,
            update_every=int(update_every),
            num_updates=int(num_updates)
        )

    optimizer = BayesianOptimization(
        f=train_func,
        pbounds=pbounds,
        random_state=1,
    )

    # New optimizer is loaded with previously seen points
    if os.path.exists(logs):
        load_logs(optimizer, logs=[logs])

    # log new points
    logger = JSONLogger(path=logs)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=100,
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(f"Max Value: {optimizer.max}")


if __name__ == '__main__':
    args = parser.parse_args()
    parameter_search(args.env, args.curve, args.logs)
    logger.info(f'Done - elapsed time: {time.process_time()} seconds')
