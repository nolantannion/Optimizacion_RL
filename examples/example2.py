'''
Script that calculates and represents various trajectories obtained by the agent in a potential never seen during the training
'''

import numpy as np
from rl_opt import PotentialEnv, find_path, load_model
import matplotlib.pyplot as plt

np.random.seed(7)

def potential(x, y):
    return (
        0.4*np.sin(5*x) * np.cos(4*y)
        + np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        - 1.2*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

def grad(x, y):
    gx = (
        0.4*5*np.cos(5*x)*np.cos(4*y)
        - (2*(x-0.2)/0.01)*np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        + 1.2*(2*(x-0.7)/0.02)*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

    gy = (
        -0.4*4*np.sin(5*x)*np.sin(4*y)
        - (2*(y-0.8)/0.01)*np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        + 1.2*(2*(y-0.3)/0.02)*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

    gr = np.array([gx,gy], dtype= np.float32)
    gr = gr/np.linalg.norm(gr+1e-8)

    return gr


env = PotentialEnv(V_fn= potential, grad_fn= grad)
model = load_model(env= env)

N = 100
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
X, Y = np.meshgrid(x,y)

Z = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        Z[i,j]  = potential(x[i], y[j])

plt.figure(figsize=(7,6))
plt.contourf(X,Y,Z, levels = 50)
for _ in range (5):
    start = np.random.random(2)
    goal = np.random.random(2)
    traj, E, state = find_path(model= model, env = env, start= start, goal = goal, view_traj= False)
    
    if state:
        plt.plot(traj[:,0], traj[:,1])
        plt.scatter([traj[0,0], traj[-1,0]], [traj[0,1], traj[-1,1]])


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectries in Potential field')
plt.colorbar(label = 'V')
plt.axis('tight')
plt.savefig('Potential_example.png', dpi = 300)
plt.show()

