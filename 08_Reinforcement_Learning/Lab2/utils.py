import numpy as np


def avg_reward(scores):
    """
    Suaviza la curva de convergencia
    """
    episode_number = np.linspace(1, len(scores) + 1, len(scores) + 1)
    acumulated_rewards = np.cumsum(scores)

    return [acumulated_rewards[i] / episode_number[i]
            for i in range(len(acumulated_rewards))]
