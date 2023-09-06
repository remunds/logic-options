# Object-Centric Option-Critic [WIP]
This repository implements the combination of the **Option-Critic** framework ([Bacon et al., 2016](https://arxiv.org/abs/1609.05140)) with the **Object-Centric** paradigm using PyTorch.

The idea is to apply both **temporal abstraction** (options) with **state abstraction** (object-centricity) to create **eXplainable Reinforcement Learning (XRL)** agents.

## Features
* Multi-level option hierarchy consisting of SB3 models trained with PPO
* SCOBI: object-centric observation input, optionally transformed into an interpretable concept bottleneck
* Supports `gymnasium` Atari environments with all respective features
* Options: define number of hierarchy levels and number of option per level individually, regularize options length and options entropy
* Control model's action selection temperature
* Parameter scheduling
* Specify experiment parameters via YAML and experiment queueing

### More to come...
* Reward shaping
* Reward cropping

## Issues
* Implicit float32-to-float64 conversions (and vice versa) cause floating point imprecision, may affect model training performance (to be investigated)

## Requirements
```
torch>=2.0.1
tensorboard>=2.13.0
gymnasium>=0.28.1
ocatari
```

## Acknowledgements
Thanks to [Laurens Weitkamp](https://github.com/lweitkamp) for the PyTorch implementation of the Option-Critic framework.
