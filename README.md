# MDP & MC Toolkit

This repository contains tools for working with **Markov Chains (MC)** and **Markov Decision Processes (MDP)**. It includes methods for visualization, simulation, model checking, and reinforcement learning.

---

## 📦 Environment Setup

We recommend using `conda` to manage dependencies and isolate the environment.

### 1. Create a new environment

```bash
conda create -n mdp_env python=3.10
conda activate mdp_env
```

### 2. Install dependencies

Install core packages from conda:

```bash
conda install -c conda-forge cvxpy matplotlib numpy
```

Install other packages via pip (e.g. `pymdptoolbox` is not available on conda):

```bash
pip install pymdptoolbox imageio
```

or:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Overview

| File | Description |
|------|-------------|
| `visu.py` | Visualization tool for MDPs and Markov Chains. Helps generate Graph for the state transitions. |
| `simulation.py` | Simulates the evolution of a Markov Chain over time. Also generates a **GIF animation** showing how the system's state changes. |
| `model_checking.py` | Contains model checking techniques for **Markov Chains (MC)** including:<br>• Symbolic methods<br>• Iterative techniques<br>• Monte Carlo simulation<br>• Sequential Probability Ratio Test (SPRT) |
| `mdp_model_checking.py` | Performs model checking specifically for **Markov Decision Processes (MDP)**. |
| `renforcment.py` | Implements **Reinforcement Learning** to find optimal policies for MDPs using techniques such as value iteration or Q-learning. |

---



