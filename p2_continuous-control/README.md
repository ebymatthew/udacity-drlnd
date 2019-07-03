[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

This project contains code for training a Deep Deterministic Policy Gradient (DDPG) agent to control a reacher arm that is tracking a positional target.  

![Trained Agent][image1]

This readme contains describes how to install the environment, train an agent and run a trained agent.
Refer to the [report.md](report.md) for an in depth description of the implementation and how DDPGs work.

### Getting Started

1. Install Anaconda (or Miniconda): https://docs.anaconda.com/anaconda/install/

1. Create and activate a conda environment following the instructions on the [Udacity Deep Reinforcment Learning Repo](https://github.com/udacity/deep-reinforcement-learning#dependencies)

1. Download and unzip the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Reacher Environment: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

1. Next you'll need to edit determine the path of the unzipped Reacher Unity environment you downloaded above. You'll pass the path to the scripts described in the following sections:
    
    - **Mac**: `"path/to/Reacher.app"`
    - **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
    - **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
    - **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
    - **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
    - **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
    - **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

### Training 

If you'd like to train a new agent run the train script:
 
 ` python train.py --env ./Reacher.app --episodes 300`
 
If you've configured the a Unity environment with visualization you can watch the agent as it's training.