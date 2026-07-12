# Basic use
import numpy as np

# Gym
import gymnasium as gym
from gymnasium import spaces


# Gym env for the model
class PotentialEnv(gym.Env):
    '''
    INPUTS:
    - V_fn: function that represents the potential
    - grad_fn: function that represents the gradient of the potential
    - speed: movement speed of the agent. Default 0.25
    - dt: time interval between 2 steps. Default 0.02
    - max_steps: max of steps in an iteration. Default 350

    ACTION:
    The posible actions are in a continuous space that selects an angle between -pi an pi in an array of shape (1,).
    \n Format: action = [angle]


    FUNCTIONS:

    - _obs: returns an array -> [pos, dir_to_goal, V, dV] as a np float 32
    - reset: inputs a initial position and a goal and returns _obs and {}
    - step: inputs an adequate action and returns _obs(), reward, terminated, truncated, self.energy
    '''
    def __init__(self, V_fn, grad_fn, speed = 0.25, dt = 0.02, max_steps=350, weights=None):
        super().__init__()

        self.V_fn = V_fn
        self.grad_fn = grad_fn

        # Parametros del sistema
        self.max_steps = max_steps
        self.dt = dt
        self.speed = speed

        self.energy = 0.0
        self.path_length = 0.0
        self.start_pos = np.zeros(2)

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, -1.0, -1.0, -2.0, -1.0, -1.0],
                dtype=np.float32
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.weights = weights or {
            "progress": 9.5,
            "align": 1.2,
            "slope": 0.02,
            "potential": 0.05,
            "step": 0.01
        }


    def _obs(self):
        V    = float(np.clip(self.V_fn(self.pos[0], self.pos[1]), -2, 2))
        grad = self.grad_fn(self.pos[0], self.pos[1])

        # Normalized direction vector 
        delta     = self.goal - self.pos
        len      = np.linalg.norm(delta) + 1e-8
        dir_goal  = (delta / len).astype(np.float32)


        return np.concatenate([
            self.pos,       # [0, 1]
            dir_goal,       # [-1, 1]
            [V],            # [-2, 2]
            grad            # [-1, 1]
        ]).astype(np.float32)


    def reset(self, start, goal):
        '''
        INPUTS:

        - start: starting position coordinates in format (x,y)
        - goal: tarjet coordinates in format (x,y)

        RETURNS:
        - obs
        - {}

        '''
        super().reset()

        self.pos = np.array(start, dtype= np.float32)
        self.goal = np.array(goal, dtype= np.float32)

        self.start_pos = self.pos.copy()

        self.steps = 0
        self.energy = 0.0
        self.path_length = 0.0

        return self._obs(), {}
    

    
    def step(self, action):
        # Agent selects direction of movement
        theta     = float(action[0] * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        prev_pos  = self.pos.copy()

        # Position and steps update
        self.pos  = prev_pos + self.speed * direction * self.dt
        self.steps += 1

        # Distance to the goal and previous distance
        len      = np.linalg.norm(self.goal - self.pos)
        prev_len = np.linalg.norm(self.goal - prev_pos)

        # Check for progress
        r_progress = prev_len - len

        # Calculate direction, alignment, energy and slope
        goal_dir   = (self.goal - prev_pos) / (prev_len + 1e-8)
        r_align    = float(np.dot(direction, goal_dir))

        V    = float(np.clip(self.V_fn(self.pos[0], self.pos[1]), -2, 2))
        grad = self.grad_fn(self.pos[0], self.pos[1])

        r_slope = float(-np.dot(direction, grad))

        self.energy      += V
        self.path_length += self.speed * self.dt


        # Reward of the actual step
        w = self.weights
        reward = (
              w["progress"]  * r_progress
            + w["align"]     * r_align
            + w["slope"]     * r_slope
            - w["potential"] * V
            - w["step"]
        )

        # Check if the model went out of bounds
        out_lim = np.any((self.pos < 0.0) | (self.pos > 1.0))
        if out_lim:
            reward -= 2.0

        # Check if the model reached the objective
        terminated = len < 0.02
        if terminated:
            reward += 20.0

        # Check if the model exceeded max steps
        truncated = self.steps >= self.max_steps

        return self._obs(), reward, terminated, truncated, self.energy