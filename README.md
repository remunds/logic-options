# Object-Centric Option-Critic [WIP]
This repository implements the combination of the **Option-Critic** framework ([Bacon et al., 2016](https://arxiv.org/abs/1609.05140)) with the **Object-Centric** paradigm using PyTorch.

The idea is to apply both **temporal abstraction** (options) with **state abstraction** (object-centricity) to create **eXplainable Reinforcement Learning (XRL)** agents.

## Features
* Supports `gymnasium` Atari environments with all respective features (framestacking, frameskipping etc.)
* Options: define number of options, regularize options length and options entropy
* Control model's action selection temperature
* Parameter scheduling
* Replay buffer
* Specify experiment parameters via YAML file

### More to come...
* Higher-level options hierarchy (currently only one options level supported)
* PPO2
* Reward shaping
* Reward cropping

## Issues
* Implicit float32-to-float64 conversions (and vice versa) cause floating point imprecision, may affect model training performance

## Requirements
```
torch>=2.0.1
tensorboard>=2.13.0
gymnasium>=0.28.1
ocatari
```

## Acknowledgements
Thanks to [Laurens Weitkamp](https://github.com/lweitkamp) for the PyTorch implementation of the Option-Critic framework.
