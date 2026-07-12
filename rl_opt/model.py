import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

def load_model(env=None, model_path = None):
    '''
    Loads the pre-trained SAC agent using the repo distribution

    INPUTS:

    - env: PotentialEnv that contains the desired potential and grad

    RETURNS:

    - model
    '''
    if model_path is None:
        model_path = (
        Path(__file__).parent.parent
        / "models"
        / "potential_model.zip"
    )

    model = SAC.load(model_path, env=env)

    return model



def find_path(model, env, start, goal, view_traj = True):

    '''
    INPUTS: 

    - model: SAC model that acts on the potential env 
    - env: enviroment for the model using the desired potential
    - start: array with the (x,y) coordinates of the initial position of the agent
    - goal: array with the (x,y) coordinaes of the goal position
    - view_traj: boolean, if true the trajectory will be shown. Default setting is true


    OUTPUTS:
    - traj = numpy array that contains all the positions (x,y). Array of length steps with each element corresponding to the coordinates in that step
    - energy = float with the accumulated energy of the trajectory
    '''

    obs, _ = env.reset(start, goal)
    env.start_pos = env.pos.copy()

    terminated, truncated  = False, False

    traj = [env.pos.copy()]

    while not (terminated or truncated):
        action, _  = model.predict(obs, deterministic = True)

        obs, reward, terminated, truncated, energy = env.step(action)

        traj.append(env.pos.copy())


    # If the traj could not be found, inform the user and returns none
    if truncated:
        print(f'\n The agent could not find a path \n')
        return None, None, False
    

    traj = np.array(traj, dtype = np.float32)


    if view_traj:                
        res = 150
        xs = np.linspace(0, 1, res)
        ys = np.linspace(0, 1, res)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(res):
            for j in range(res):
                Z[i, j] = env.V_fn(X[i, j], Y[i, j])

        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, Z, levels=50)
        plt.colorbar(label= "V(x,y)")
        plt.plot(traj[:, 0], traj[:, 1], color = 'r')
        plt.scatter(*start, label="Start", c = 'w')
        plt.scatter(*goal, label="Goal", c = 'k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectory')
        plt.legend()

        plt.show()


    if terminated:
        print(f'\n Trajectory found succesfully \n')
        return traj, energy, True



