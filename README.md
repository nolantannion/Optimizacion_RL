# RL Trajectory Optimization

Trajectory optimization in two-dimensional potential fields using Reinforcement Learning.

This project provides a pre-trained Soft Actor-Critic (SAC) agent capable of finding trajectories between two points while considering the local potential field and its gradient.

The main objective is to obtain energy-efficient trajectories while successfully reaching the target position.

## Overview

The agent moves inside a normalized two-dimensional domain:

```text
x ∈ [0, 1]
y ∈ [0, 1]
```

At every step, the SAC policy selects the direction of movement of the agent.

The observation received by the model is:

```text
[x, y, dx_goal, dy_goal, V, gx, gy]
```

where:

* `x`, `y`: current agent position.
* `dx_goal`, `dy_goal`: normalized direction towards the target.
* `V`: local potential value.
* `gx`, `gy`: local potential gradient.

The action is a scalar in `[-1, 1]`, mapped to an angle in `[-π, π]`.

The model was trained on procedurally generated potential fields and evaluated on fields not used during training.

## Evaluation

The model was evaluated on potential fields not used during training. To reduce the dependence of individual trajectories or performance cases, evaluation is performed in 3 different potential fields until we obtained 150 succesful trajectories in each one. 

Final metrics are obtained averaging all the results, being the uncertainity the standard deviation of the data.

### Evaluation Potential Fields

The potentials used to evaluate are the following.

#### Potential V0

$V_0(x,y) = 0.4\sin(5x)\cos(4y)+\exp\left(-\frac{(x-0.2)^2+(y-0.8)^2}{0.01}\right)-1.2\exp\left(-\frac{(x-0.7)^2+(y-0.3)^2}{0.02}\right)$

#### Potential V1

$V_1(x,y) = -\exp\left(-\frac{(x-0.3)^2+(y-0.3)^2}{0.12}\right)-\exp\left(-\frac{(x-0.7)^2+(y-0.7)^2}{0.12}\right)$

#### Potential V2

$V_2(x,y)=0.3\sin(4x)\sin(3y)-\exp\left(-\frac{(x-0.6)^2+(y-0.6)^2}{0.03}\right)+0.5\exp\left(-\frac{(x-0.3)^2+(y-0.2)^2}{0.02}\right)$

### Results 
The evaluation baseline is a comparison with the trivial linear trajectory that connects the start and goal points. The terms used to evaluate are:

**Success rate**: $91 \pm  2  $% of trajectories reach the target. \
**Improvement Rate**: $56 \pm 6 $% of the successful trajectories improve energy consuption over the baseline.

The scripts used to reproduce these measurements are available in the `evaluation/` directory.



## Repository Structure

```text
Optimizacion_RL/
│
├── rl_opt/
│   ├── __init__.py
│   ├── env.py
│   └── model.py
│
├── models/
│   └── sac_model.zip
│
├── examples/
│   └── basic_use.py
│
├── evaluation/
│   ├── evaluate_model.py
│   └── metrics.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

### `rl_opt`

Contains the main reusable code of the project.

* `env.py`: implementation of the `PotentialEnv` Gymnasium environment.
* `model.py`: utilities for loading the pre-trained SAC agent and computing trajectories.
* `__init__.py`: exposes the public project interface.

### `models`

Contains the pre-trained SAC model used by the project.

### `examples`

Contains minimal usage examples.

### `evaluation`

Contains the scripts used to evaluate the model and obtain the metrics reported.

The evaluation code is included to support and reproduce the reported results. Its main purpose is to provide transparency and credibility to the project.

As this code was originally developed for internal evaluation rather than public reuse, some sections and comments remain in Spanish, the author's native language, and have not been extensively refactored or rewritten.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Optimizacion_RL
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

Import the environment and model utilities:

```python
from rl_opt import PotentialEnv, load_model, find_path
```

Define a potential function and its gradient:

```python
import numpy as np


def potential(x, y):
    return np.sin(2 * np.pi * x)


def gradient(x, y):
    return np.array([2 * np.pi * np.cos(2 * np.pi * x), 0.0], dtype=np.float32)
```

Create the environment and load the pre-trained model:

```python
env = PotentialEnv(potential, gradient)

model = load_model(env)
```

Define the initial and target positions:

```python
start = np.array([0.1, 0.2])
goal = np.array([0.8, 0.9])
```

Compute the trajectory:

```python
trajectory = find_path(
    model=model,
    env=env,
    start=start,
    goal=goal
)
```

The returned trajectory is a NumPy array with shape:

```text
(n_steps, 2)
```

where each row contains the `(x, y)` position of the agent at a given step.

A complete example is available in:

```text
examples/basic_use.py
```

It can be executed from the repository root using:

```bash
python -m examples.basic_use
```
**For ideal functioning user should execute the terminal command with the corresponding .venv activated and being located in the Optimizacion_RL folder**

## Environment

`PotentialEnv` is based on the Gymnasium API.

The environment requires two callable functions:

```python
PotentialEnv(V_fn, grad_fn)
```

### Potential function

The potential must accept two coordinates:

```python
V_fn(x, y)
```

and return a scalar potential value.

### Gradient function

The gradient must accept the same coordinates:

```python
grad_fn(x, y)
```

and return a two-dimensional vector:

```text
[gx, gy]
```

The input fields should follow the same numerical scale and normalization used by the model observation space. Large differences from the training distribution may reduce policy performance.

## Reinforcement Learning Model

The agent uses Soft Actor-Critic (SAC), an off-policy reinforcement learning algorithm designed for continuous action spaces.

The reward function combines:

* Progress towards the goal.
* Alignment with the target direction.
* Local potential gradient.
* Potential energy.
* Step penalty.

The trained policy therefore balances direct progress towards the target with the energetic cost associated with the potential landscape.



## Limitations

The model operates in a normalized two-dimensional domain under constant movement speed.

The policy only receives local information about the potential and its gradient. It does not receive a complete map of the potential field.

As with most reinforcement learning models, performance may decrease for potential landscapes significantly outside the distribution encountered during training.

The model is more a proof of concept of the trajectory optimization opportunities that exist with RL rather than a final model that can be used in production or more serious path finding contexts.  

## Dependencies

The main dependencies are:

* Python
* NumPy
* Gymnasium
* Stable-Baselines3
* PyTorch
* Matplotlib

See `requirements.txt` for the complete dependency list.

## License

This project is distributed under the terms specified in the `LICENSE` file.
