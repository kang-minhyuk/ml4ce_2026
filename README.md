# ML4CE 2026 Coursework

This repository contains the coursework for **Machine Learning for Chemical
Engineering**:

- [Bayesian Optimisation](./BO_coursework/)
- [Reinforcement Learning](./RL_coursework/)

## Start here: Bayesian Optimisation

The BO coursework is the recommended starting point.

```bash
cd BO_coursework
uv sync
uv run jupyter lab student/project.ipynb
```

Work in `student/project.ipynb`. The notebook explains the task and writes
your algorithm to `algorithms/my_algorithm.py`.

## Reinforcement Learning

To work on the RL coursework, set up its environment separately:

```bash
cd RL_coursework
uv sync
uv run jupyter lab RL_coursework.ipynb
```

See each coursework directory for its own files and instructions.
