# Tennis

![DDPG Tennis](assets/unity_reacher_ppo_agent.gif)

The environment is composed of two agents each one controlling a racket to bounce a ball over a net. 

- If an agent hits the ball over the net, it receives a reward of +0.1. 
- If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 

Therefore, the overall episode reward is maximized if the ball is kept in play by both agents

Each timestep, each agent recieves:

- A 24 element long vector representing the current state

The actions space is continuous and consists of the movement toward net or away from net, and jumping.

Finally, the environment is considered solved when the average score of the last 100 episodes is greater than +0.5.

The episode score is calculated by adding up the rewards that each agent received (without discounting) then taking the maximum of these two scores.

## Training 

To train the agent simply run `python train_tennis_ddpg.py`. All hyperparameters can be modified within the script file.   

## Results 

A [trained model](saved_models/agent_ddpg.ckpt) with an average score of XX over 100 episodes is included in this repository.

For a more complete description of the results, refer to the [report](report.md) page.

To visualise the trained agent run:

```
python watch_trained_tennis.py --agent data/ppo.ckpt
``` 

## Installation

Create a new Python 3.6 environment.

```
conda create --name tennis python=3.6 
activate tennis
```

Install ml-agents using the repository.

```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
git checkout 0.4.0b
cd python 
pip install .
```

Install PyTorch using the recommended [pip command](https://pytorch.org/) from the PyTorch site. For example, to install with CUDA 9.2: 

```
conda install pytorch cuda92 -c pytorch
```

Clone this repository locally. 

```
git clone https://github.com/ostamand/continuous-control.git
```

Finally, download the environment which corresponds to your operationg system. Copy/paste the extracted content to the `data` subfolder. 

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)












