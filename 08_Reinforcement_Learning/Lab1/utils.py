import itertools
import numpy as np
import matplotlib.pyplot as plt


def show_learning_curve(agent):
    # Se muestra la curva de aprendizaje de los pasos por episodio
    episode_steps = np.array(agent.timesteps_of_episode)

    plt.figure(figsize=(20, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.array(range(0, len(episode_steps))), episode_steps)
    ax1.set_title('Pasos (timesteps) por episodio')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Timesteps')

    # Se suaviza la curva de aprendizaje
    episode_number = np.linspace(
                        1, len(episode_steps) + 1, len(episode_steps) + 1
                    )
    acumulated_steps = np.cumsum(episode_steps)

    steps_per_episode = [acumulated_steps[i] / episode_number[i]
                         for i in range(len(acumulated_steps))]

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(steps_per_episode)
    ax2.set_title('Pasos (timesteps) acumulados por episodio')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accumulated Timesteps')

    plt.show()


def show_reward_curve(agent):
    # Se muestra la curva de convergencia de las recompensas
    episode_rewards = np.array(agent.reward_of_episode)
    # plt.scatter(
    #     np.array(range(0, len(episode_rewards))),
    #     episode_rewards,
    #     s=0.7
    # )
    # plt.title('Recompensa por episodio')
    # plt.show()

    # Se suaviza la curva de convergencia
    episode_number = np.linspace(
                        1, len(episode_rewards) + 1, len(episode_rewards) + 1
                    )
    acumulated_rewards = np.cumsum(episode_rewards)

    reward_per_episode = [acumulated_rewards[i] / episode_number[i]
                          for i in range(len(acumulated_rewards))]

    plt.figure(figsize=(10, 3))
    plt.plot(reward_per_episode)
    plt.title('Recompensa acumulada por episodio')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    plt.show()


def show_value_matrix(agent):
    """
    Se procede con los cálculos previos a la graficación de la matriz de valor
    """
    value_matrix = np.zeros((4, 4))
    for row in range(4):
        for column in range(4):
            state_values = [agent.q.get((row * 4 + column, action), 0)
                            for action in range(4)]

            """
            Como usamos epsilon-greedy, determinamos la acción que
            arroja máximo valor
            """
            maximum_value = max(state_values)

            # Removemos el ítem asociado con la acción de máximo valor
            state_values.remove(maximum_value)

            """
            El valor de la matriz para la mejor acción es el máximo valor
            por la probabilidad de que el mismo sea elegido (que es 1-epsilon
            por la probabilidad de explotación más 1/4 * epsilon por
            probabilidad de que sea elegido al azar cuando se opta por una
            acción exploratoria)
            """
            value_matrix[row, column] = maximum_value * (1 - agent._epsilon + 1/4 * agent._epsilon)

            for non_maximum_value in state_values:
                value_matrix[row, column] += agent._epsilon/4 * non_maximum_value

    """
    El valor del estado objetivo se asigna en 1 (reward recibido al llegar)
    para que se coloree de forma apropiada
    """
    value_matrix[3, 3] = 1

    # Se grafica la matriz de valor
    plt.figure(figsize=(10, 3))
    plt.imshow(value_matrix, cmap=plt.cm.RdYlGn)
    plt.tight_layout()
    plt.colorbar()

    # fmt = '.2f'
    # thresh = value_matrix.max() / 2.

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
