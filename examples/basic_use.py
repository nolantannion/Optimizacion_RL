import numpy as np
from rl_opt import PotentialEnv, find_path, load_model



def potential(x,y):
    return x**2 - y**2

def grad(x, y):
    gx = 2*x
    gy = -2*y 
    grad = np.array([gx, gy], dtype=np.float32)
    return grad/(np.linalg.norm(grad) +1e-8)


env = PotentialEnv(V_fn= potential, grad_fn= grad)
model = load_model(env= env)

start = [0.1, 0.2]
goal = [0.8, 0.9]

find_path(model= model, env = env, start= start, goal = goal)