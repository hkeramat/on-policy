# MAPPO

Chao Yu*, Akash Velu*, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. 

This repository implements MAPPO, a multi-agent variant of PPO. The implementation in this repositorory is used in the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955). This repository is heavily based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

<font color="red"> All hyperparameters and training curves are reported in appendix, we would strongly suggest to double check the important factors before runing the code, such as the rollout threads, episode length, ppo epoch, mini-batches, clip term and so on. <font color='red'>Besides, we have updated the newest results on google football testbed and suggestions about the episode length and parameter-sharing in appendix, welcome to check that. </font>

<font color="red"> We have recently noticed that a lot of papers do not reproduce the mappo results correctly, probably due to the rough hyper-parameters description. We have updated training scripts for each map or scenario in /train/train_xxx_scripts/*.sh. Feel free to try that.</font>
 

## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)
- [Google Research Football (GRF)](https://github.com/google-research/football)

## 1. Usage
**WARNING: by default all experiments assume a shared policy by all agents i.e. there is one neural network shared by all agents**

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code
for MAPPO. 

* The envs/ subfolder contains environment wrapper implementations for the MPEs, SMAC, and Hanabi. 

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name (in the case of SMAC and the MPEs) can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.




