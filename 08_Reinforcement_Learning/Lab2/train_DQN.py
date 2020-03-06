#!/users/mferreyra/.virtualenvs/diplodatos-rl/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import CartPole_SGD
import CartPole_SGD_Poly
import CartPole_SGD_Poly_Dual
import CartPole_DQN
from utils import avg_reward


if __name__ == '__main__':

    sns.set_style('whitegrid')
    sns.set_context('talk')

    # CartPole: Aproximación con un modelo lineal
    print("CartPole: Aproximación con un modelo lineal")
    # agent = CartPole_SGD.SGDCartPoleSolver()
    # scores_SGD = agent.run()

    # CartPole: Aproximación con un modelo lineal con 'feature construction'
    print("CartPole: Aproximación con un "
          "modelo lineal con 'feature construction'")
    # agent = CartPole_SGD_Poly.SGDPolyCartPoleSolver()
    # scores_SGD_Poly = agent.run()

    # CartPole: Aproximación con un modelo lineal y 'Modelo Duplicado'
    print("CartPole: Aproximación con un modelo lineal y 'Modelo Duplicado'")
    # agent = CartPole_SGD_Poly_Dual.SGDPolyDualCartPoleSolver()
    # scores_SGD_Poly_Dual = agent.run()

    # CartPole: Aproximación con Redes Neuronales
    print("CartPole: Aproximación con Redes Neuronales")
    agent = CartPole_DQN.DQNCartPoleSolver()
    scores_DQN = agent.run()

    print("Start Plots ...")
    plt.figure(figsize=(25, 8))

    # Reward/Score optenido por Episodio
    ax1 = plt.subplot(1, 2, 1)
    # ax1.plot(
    #     np.array(range(0, len(scores_SGD))),
    #     np.array(scores_SGD),
    #     label='SGD',
    #     c='#5c8cbc'
    # )
    # ax1.plot(
    #     np.array(range(0, len(scores_SGD_Poly))),
    #     np.array(scores_SGD_Poly),
    #     label='SGD_Poly',
    #     c='#faa76c'
    # )
    # ax1.plot(
    #     np.array(range(0, len(scores_SGD_Poly_Dual))),
    #     np.array(scores_SGD_Poly_Dual),
    #     label='SGD_Poly_Dual',
    #     c='#71bc78'
    # )
    ax1.plot(
        np.array(range(0, len(scores_DQN))),
        np.array(scores_DQN),
        label='DQN',
        c='#7e5fa4'
    )
    ax1.set_ylim(0, 200)
    ax1.set_title('Recompensa por episodio')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='upper left')

    # Curva de Convergencia suavizada
    # reward_per_episode_SGD = avg_reward(scores_SGD)
    # reward_per_episode_SGD_Poly = avg_reward(scores_SGD_Poly)
    # reward_per_episode_SGD_Poly_Dual = avg_reward(scores_SGD_Poly_Dual)
    reward_per_episode_DQN = avg_reward(scores_DQN)

    ax2 = plt.subplot(1, 2, 2)
    # ax2.plot(
    #     reward_per_episode_SGD,
    #     label='SGD',
    #     c='#5c8cbc'
    # )
    # ax2.plot(
    #     reward_per_episode_SGD_Poly,
    #     label='SGD_Poly',
    #     c='#faa76c'
    # )
    # ax2.plot(
    #     reward_per_episode_SGD_Poly_Dual,
    #     label='SGD_Poly_Dual',
    #     c='#71bc78'
    # )
    ax2.plot(
        reward_per_episode_DQN,
        label='DQN',
        c='#7e5fa4'
    )
    ax2.set_ylim(0, 200)
    ax2.set_title('Recompensa promedio por episodio')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.legend(loc='upper left')

    # plt.savefig('/users/mferreyra/curves_rl.png')
    plt.savefig('/home/mferreyra/Desktop/DiploDatos2018/Aprendizaje_Refuerzo/Lab2/curves_rl.png')
    print("Finished Plots ...")
