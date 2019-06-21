[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project contains code for training a Deep Q-Network agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

This readme contains describes how to install the environment, train an agent and run a trained agent.
Refer to the [report.md](report.md) for an in depth description of the implementation and how Deep Q-Networks work.

### Getting Started

1. Install Anaconda (or Miniconda): https://docs.anaconda.com/anaconda/install/

1. Create and activate a conda environment following the instructions on the [Udacity Deep Reinforcment Learning Repo](https://github.com/udacity/deep-reinforcement-learning#dependencies)

1. Download and unzip the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

1. In the conda environment start Jupyter: `juypter notebook`
 
1. In Juypter open `p1_navigation/Navigation.ipynb`. You'll see saved output from a previous run of the notebook where the agent was trained.

1. Next you'll need to edit `p1_navigation/Navigation.ipynb` to use the Banana Unity environment you downloaded above.
You'll need to edit the cell which contains:
    
    `env = UnityEnvironment(file_name="Banana.app")`
    
    with the appropriate `file_name` for the environment you downloaded:
    
    - **Mac**: `"path/to/Banana.app"`
    - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
    - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
    - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
    - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
    - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
    - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`
    
    For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
    ```
    env = UnityEnvironment(file_name="Banana.app")
    ```

### Training 

If you'd like to train a new agent open `p1_navigation/Navigation.ipynb` and run all cells. If you've configured
the a Unity environment with visualization you can watch the agent as it's training and then see the trained agent run at the end.

### Watching Pre Trained Agent

If you'd like to simply watch the pre-trained agent. Open `p1_navigation/Navigation.inference.ipynb` and run all cells.