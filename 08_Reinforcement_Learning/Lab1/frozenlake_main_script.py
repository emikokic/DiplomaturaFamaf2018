#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from agents.frozen_lake_agent import FrozenLakeAgent as fP
import FrozenLakeAgent as fP

sns.set_style('whitegrid')
sns.set_context('talk')

# Definimos sus híper-parámetros básicos
alpha = 0.5
gamma = 0.9
epsilon = 0.1
tau = 25
is_slippery = False


# Se declara una semilla aleatoria
random_state = np.random.RandomState(47)

# El tiempo de corte del agente son 100 time-steps
cutoff_time = 100

# Instanciamos nuestro agente
agent = fP.FrozenLakeAgent()

hyper_parameters = {
    'alpha': alpha,
    'gamma': gamma,
    'epsilon': epsilon,
    'tau': tau,
}
agent.set_hyper_parameters(hyper_parameters)

agent.random_state = random_state

"""
Declaramos como True la variable de mostrar video, para ver en tiempo real
cómo aprende el agente.
Borrar esta línea para acelerar la velocidad del aprendizaje
"""
agent.display_video = True

# Establece el tiempo de corte del agente
agent.set_cutoff_time(cutoff_time)


"""
Inicializa el agente
slippery es establecido en False por defecto
"""
agent.init_agent(is_slippery=is_slippery)

# Reinicializa el conocimiento del agente
agent.restart_agent_learning()

# Se realiza la ejecución del agente
# avg_steps_per_episode = agent.run_q_learning(policy='epsilon-greedy')
avg_steps_per_episode = agent.run_q_learning(policy='softmax')
# avg_steps_per_episode = agent.run_sarsa()

# Se muestra la curva de convergencia de las recompensas
episode_rewards = np.array(agent.reward_of_episode)
plt.scatter(np.array(range(0, len(episode_rewards))), episode_rewards, s=0.7)
plt.title('Recompensa por episodio')
plt.show()

# Se suaviza la curva de convergencia
episode_number = np.linspace(
                    1, len(episode_rewards) + 1, len(episode_rewards) + 1
                )
acumulated_rewards = np.cumsum(episode_rewards)

reward_per_episode = [acumulated_rewards[i] / episode_number[i]
                      for i in range(len(acumulated_rewards))]

plt.plot(reward_per_episode)
plt.title('Recompensa acumulada por episodio')
plt.show()

# ---

# Se muestra la curva de aprendizaje de los pasos por episodio
episode_steps = np.array(agent.timesteps_of_episode)
plt.plot(np.array(range(0, len(episode_steps))), episode_steps)
plt.title('Pasos (timesteps) por episodio')
plt.show()

# Se suaviza la curva de aprendizaje
episode_number = np.linspace(
                    1, len(episode_steps) + 1, len(episode_steps) + 1
                )
acumulated_steps = np.cumsum(episode_steps)

steps_per_episode = [acumulated_steps[i] / episode_number[i]
                     for i in range(len(acumulated_steps))]

plt.plot(steps_per_episode)
plt.title('Pasos (timesteps) acumulados por episodio')
plt.show()

# ---

# Se procede con los cálculos previos a la graficación de la matriz de valor
value_matrix = np.zeros((4, 4))
for row in range(4):
    for column in range(4):
        # state_values = []
        # for action in range(4):
        #     state_values.append(agent.q.get((row * 4 + column, action), 0))
        state_values = [agent.q.get((row * 4 + column, action), 0)
                        for action in range(4)]

        """
        Como usamos epsilon-greedy,
        determinamos la acción que arroja máximo valor
        """
        maximum_value = max(state_values)

        # Removemos el ítem asociado con la acción de máximo valor
        state_values.remove(maximum_value)

        """
        El valor de la matriz para la mejor acción es el máximo valor por la
        probabilidad de que el mismo sea elegido (que es 1-epsilon por la
        probabilidad de explotación más 1/4 * epsilon por probabilidad de que
        sea elegido al azar cuando se opta por una acción exploratoria)
        """
        value_matrix[row, column] = maximum_value * (1 - epsilon + 1/4 * epsilon)
        for non_maximum_value in state_values:
            value_matrix[row, column] += epsilon/4 * non_maximum_value

"""
El valor del estado objetivo se asigna en 1 (reward recibido al llegar)
para que se coloree de forma apropiada
"""
value_matrix[3, 3] = 1

# Se grafica la matriz de valor
plt.imshow(value_matrix, cmap=plt.cm.RdYlGn)
plt.tight_layout()
plt.colorbar()

fmt = '.2f'
thresh = value_matrix.max() / 2.

for row, column in itertools.product(range(value_matrix.shape[0]),
                                     range(value_matrix.shape[1])):
    left_action = agent.q.get((row * 4 + column, 0), 0)
    down_action = agent.q.get((row * 4 + column, 1), 0)
    right_action = agent.q.get((row * 4 + column, 2), 0)
    up_action = agent.q.get((row * 4 + column, 3), 0)

    arrow_direction = '↓'
    best_action = down_action

    if best_action < right_action:
        arrow_direction = '→'
        best_action = right_action
    if best_action < left_action:
        arrow_direction = '←'
        best_action = left_action
    if best_action < up_action:
        arrow_direction = '↑'
        best_action = up_action
    if best_action == 0:
        arrow_direction = ''

    """
    Notar que column, row están invertidos en orden en la línea de abajo
    porque representan a x,y del plot
    """
    plt.text(column, row, arrow_direction, horizontalalignment="center")

plt.xticks([])
plt.yticks([])
plt.show()

print('\n Matriz de valor (en números): \n\n', value_matrix)

agent.destroy_agent()
