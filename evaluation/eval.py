'''
This script evaluates the model on potential fields that were not used during training.

The evaluation code is included to support and reproduce the metrics reported in the README. The main purpose of this section is to provide transparency and credibility to the reported results.

As this code was originally developed for internal evaluation rather than public reuse, some sections and comments remain in Spanish, the author's native language, and have not been extensively rewritten.

'''
# Uso basico
import numpy as np
import matplotlib.pyplot as plt

# Gym
import gymnasium as gym
from gymnasium import spaces

# Stable baselines
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


# Sistema y rutas
from pathlib import Path



# Gaussianas
def gaussianas_gradiente(n=3):
    '''
    Funcion que devuelve funciones de potencial con n gaussianas y sus gradientes
    '''
    centers = np.random.uniform(0, 1, size=(n, 2))
    amps = np.random.uniform(0.5, 1.5, size=n)
    sigmas = np.random.uniform(0.05, 0.15, size=n)

    def V(x, y):
        dx = x - centers[:,0]
        dy = y - centers[:,1]
        r2 = dx*dx + dy*dy
        return -np.sum(amps * np.exp(-r2/(2*sigmas**2)))

    def gradV(x, y):
        dx = x - centers[:,0]
        dy = y - centers[:,1]
        r2 = dx*dx + dy*dy

        exp_term = np.exp(-r2/(2*sigmas**2))
        coeff = amps * exp_term / (sigmas**2)

        gx = np.sum(coeff * dx)
        gy = np.sum(coeff * dy)

        grad = np.array([gx, gy], dtype=np.float32)
        return grad / (np.linalg.norm(grad) + 1e-8)

    return V, gradV


# Seno/coseno
def sin_cos_grad():
    '''
    Funcion equivalente con senos y cosenos
    '''
    def V(x, y):
        return np.sin(3*x)*np.cos(3*y)

    def gradV(x, y):
        gx = 3*np.cos(3*x)*np.cos(3*y)
        gy = -3*np.sin(3*x)*np.sin(3*y)
        grad = np.array([gx, gy], dtype=np.float32)
        return grad / (np.linalg.norm(grad) + 1e-8)

    return V, gradV


# sampler
def sample_potential():
    '''
    Funcion que elige un potencial aleatorio para pasar al entorno
    '''
    if np.random.rand() < 0.5:
        return gaussianas_gradiente()
    else:
        return sin_cos_grad()


class FastGradEnv(gym.Env):
    def __init__(self, max_steps=350, weights=None):
        super().__init__()

        # Parametros del sistema
        self.max_steps = max_steps
        self.dt = 0.02
        self.speed = 0.25

        self.energy = 0.0
        self.path_length = 0.0
        self.start_pos = np.zeros(2)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


        # Lo que devuelve obs [x, y, dx_goal, dy_goal, V, gx, gy]:
        #   [0] x        - posición en x          (en [0, 1])
        #   [1] y        - posición en y          (en [0, 1])
        #   [2] dx_goal  - delta x hacia el goal  (en [-1, 1])
        #   [3] dy_goal  - delta y hacia el goal  (en [-1, 1])
        #   [4] V        - potencial local         (en [-2, 2])
        #   [5] gx       - gradiente en x          (en [-1, 1])
        #   [6] gy       - gradiente en y          (en [-1, 1])
        self.observation_space = spaces.Box(
            low=np.array( [0.0,  0.0,  -1.0, -1.0, -2.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,   1.0,  1.0,  2.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )


        self.weights = weights or {
            "progress": 9.5,
            "align": 1.2,
            "slope": 0.02,
            "potential": 0.05,
            "step": 0.01
        }

    # Funcion de reinicio, elige un nuevo potencial, posiciones y goal aleatorios
    # evita exactamente situarsse en el borde de la caja
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.V_fn, self.grad_fn = sample_potential()

        self.pos  = self.np_random.uniform(0.05, 0.95, 2).astype(np.float32)
        self.goal = self.np_random.uniform(0.05, 0.95, 2).astype(np.float32)

        # Almacenamos la posicion inicial
        self.start_pos = self.pos.copy()

        self.steps = 0


        self.energy      = 0.0
        self.path_length = 0.0

        return self._obs(), {}

    def _obs(self):
        V    = float(np.clip(self.V_fn(self.pos[0], self.pos[1]), -2, 2))
        grad = self.grad_fn(self.pos[0], self.pos[1])

        # Normalizar el vector dirección al goal para que quede en [-1, 1]
        delta     = self.goal - self.pos
        len      = np.linalg.norm(delta) + 1e-8
        dir_goal  = (delta / len).astype(np.float32)


        return np.concatenate([
            self.pos,       # [0, 1]
            dir_goal,       # [-1, 1]
            [V],            # [-2, 2]
            grad            # [-1, 1]
        ]).astype(np.float32)

    def step(self, action):
        # El agente elige un angulo y se mueve en esa direccion
        # es necesario action[0] porque esta definido como un vector
        theta     = float(action[0] * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        prev_pos  = self.pos.copy()

        # Actualizamos la posicion
        self.pos  = prev_pos + self.speed * direction * self.dt
        self.steps += 1

        # lenancia a la meta y lenancia previa
        len      = np.linalg.norm(self.goal - self.pos)
        prev_len = np.linalg.norm(self.goal - prev_pos)

        # Calculamos el progreso
        r_progress = prev_len - len

        # Comprobar la direccion y alineamiento. Usamos prev porque ya hemos actualizado posiciones
        goal_dir   = (self.goal - prev_pos) / (prev_len + 1e-8)
        r_align    = float(np.dot(direction, goal_dir))

        V    = float(np.clip(self.V_fn(self.pos[0], self.pos[1]), -2, 2))
        grad = self.grad_fn(self.pos[0], self.pos[1])

        r_slope = float(-np.dot(direction, grad))

        # Añadimos la energia y longitud del paso
        self.energy      += V
        self.path_length += self.speed * self.dt


        # Funcion de reward ajustada con los pesos proporcionados o por defecto
        w = self.weights
        reward = (
              w["progress"]  * r_progress
            + w["align"]     * r_align
            + w["slope"]     * r_slope
            - w["potential"] * V
            - w["step"]
        )

        # Comprobamos si esta fuera y penalizamos
        out_lim = np.any((self.pos < 0.0) | (self.pos > 1.0))
        if out_lim:
            reward -= 2.0

        # Comprobamos si llega al destino y recompensamos
        terminated = len < 0.02
        if terminated:
            reward += 20.0

        # Comprobamos si alcanza el limite de pasos
        truncated = self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            direct_len = np.linalg.norm(self.goal - self.start_pos)
            efficiency  = direct_len / (self.path_length + 1e-8)
            info = {
                "energy":     self.energy,
                "success":    float(terminated),
                "efficiency": efficiency
            }

        return self._obs(), reward, terminated, truncated, info




'''
Potenciales para test no vistos durante el enternamiento del modelo.
Utilizados en conjunto para el calculo de las metricas del modelo
'''

def V_test(x, y):
    return (
        0.4*np.sin(5*x) * np.cos(4*y)
        + np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        - 1.2*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

def grad_test(x, y):
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

# Potencial y gradiente de test alternativo
# def V_test(x, y):
#     return (
#         -np.exp(-((x-0.3)**2 + (y-0.3)**2)/0.12)
#         -np.exp(-((x-0.7)**2 + (y-0.7)**2)/0.12)
#     )

# def grad_test(x, y):
#     f1 = np.exp(-((x-0.3)**2 + (y-0.3)**2)/0.12)
#     f2 = np.exp(-((x-0.7)**2 + (y-0.7)**2)/0.12)

#     gx = (2*(x-0.3)/0.12)*f1 + (2*(x-0.7)/0.12)*f2
#     gy = (2*(y-0.3)/0.12)*f1 + (2*(y-0.7)/0.12)*f2

#     gr = np.array([gx,gy], dtype= np.float32)
#     gr = gr/np.linalg.norm(gr+1e-8)

#     return gr

#     return np.array([gx, gy], dtype=np.float32)


# Potencial alternativo tipo silla de montar
# No es gaussiano ni sinusoidal como el entrenamiento
# def V_test(x,y):
#     return x**2 - y**2

# def grad_test(x, y):
#     gx = 2*x
#     gy = -2*y
#     grad = np.array([gx, gy], dtype=np.float32)
#     return grad/(np.linalg.norm(grad) +1e-8)




# Entorno para el test
class TestEnv(FastGradEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.V_fn = V_test
        self.grad_fn = grad_test

        # Evita casos de inicio demasiado cercano
        while True:
            self.pos = self.np_random.uniform(0.01,0.99,2)
            self.goal = self.np_random.uniform(0.01,0.99,2)

            if np.linalg.norm(self.goal - self.pos) < 0.7:
                break

        self.start_pos = self.pos.copy()

        return self._obs(), {}


# Creamos el entorno
env = TestEnv()
model_path = (Path(__file__).parent.parent/ "models"/ "potential_model.zip")
model = SAC.load(model_path, env=env)
# Seed para numeros aleatorios
seed = 12

if seed is not None:
  np.random.seed(seed)


# Parámetros
n_episodes = 100
max_trials = 500
plot_hist = True

# Metricas
rewards, energies, effs = [], [], []
linear_energies = []

exitos = 0
trials = 0
rl_better_count = 0



#  evaluación
while exitos < n_episodes and trials < max_trials:
    trials += 1

    if seed is not None:
      obs, _ = env.reset(seed + trials)

    else:
      obs, _ = env.reset()

    start = env.pos.copy()
    goal = env.goal.copy()

    traj = [start.copy()]
    total_r = 0

    done = False
    term = False

    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        action = action[0]

        obs, r, term, trunc, _ = env.step(action)

        total_r += r
        traj.append(env.pos.copy())
        done = term or trunc

    if not term:
        continue

    # -------- solo éxitos --------
    exitos += 1

    traj = np.array(traj)

    path_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    direct_dist = np.linalg.norm(start - goal)

    n_points = int(direct_dist/(env.speed*env.dt))
    xs = np.linspace(start[0], goal[0], n_points)
    ys = np.linspace(start[1], goal[1], n_points)

    energy_lin = 0.0
    for x, y in zip(xs, ys):
        energy_lin += env.V_fn(x, y)

    energy_rl = env.energy

    if energy_rl < energy_lin:
        rl_better_count += 1


    energies.append(energy_rl)
    linear_energies.append(energy_lin)

# Resultados
if exitos < n_episodes:
    print(f"Warning: solo {exitos} éxitos en {trials} intentos")


energies = np.array(energies)
linear_energies = np.array(linear_energies)

# Mostramos histograma de energias de las trayectorias
if plot_hist and exitos > 0:
    plt.figure()

    plt.hist(energies, bins=25, alpha=0.6, label="RL energy", color='r')
    plt.hist(linear_energies, bins=25, alpha=0.6, label="Linear energy", color='b')

    plt.axvline(np.mean(energies), linestyle="--", color='r', label="Mean RL")
    plt.axvline(np.mean(linear_energies), linestyle="--", color='b', label="Mean Linear")

    plt.xlabel("Energy")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de energías (solo éxitos)")
    plt.legend()
    plt.show()

# métricas finales
results = {
    "Exito de llegada": exitos / trials,
    "E RL": float(np.mean(energies)) if exitos > 0 else np.nan,
    "E lineal": float(np.mean(linear_energies)) if exitos > 0 else np.nan,
    "Tasa de mejora": rl_better_count / exitos if exitos > 0 else np.nan,
}

titulos = results.keys()

for titulo in titulos:
  print(f'{titulo}: {results[titulo]:.3f}')
