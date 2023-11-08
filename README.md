# Hierarchical Logic RL-Agents

**This repository extends logic RL agents (based on [NUDGE](https://github.com/k4ntz/NUDGE)) with the Option-Critic framework ([Bacon et al., 2016](https://arxiv.org/abs/1609.05140)) using PyTorch**.

The idea is to apply temporal abstraction (options) to improve the interpretability of logic agents, especially for more complex (real-world) tasks where the logic policy grows inscrutably large.

It is part of to the Master Thesis "Eyeing the Big Play, Not Just the Moves: Advancing the Interpretability of RL Agents through Temporal Abstraction via Options."

## Features
* Multi-level option hierarchy consisting of SB3 models trained with PPO
* SCOBI: object-centric observation input, optionally transformed into an interpretable concept bottleneck
* Options: define number of hierarchy levels and number of option per level individually, regularize options length and options entropy
* Parameter scheduling
* Hyperparameter configuration via YAML

## Requirements
```
torch>=2.0.1
tensorboard>=2.13.0
gymnasium>=0.28.1
ocatari
scobi
nudge
```

## Acknowledgements
Thanks to [Laurens Weitkamp](https://github.com/lweitkamp) for the PyTorch implementation of the Option-Critic framework.
