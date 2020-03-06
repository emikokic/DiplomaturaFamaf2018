import gym
import six
import numpy as np
from gym.envs.registration import register


class FrozenLakeAgent:

    def __init__(self):
        # Basic Configuration
        self._environment_name = "FrozenLake-v0"

        # NOTE: that the "None" variables have values yet to be assigned
        self._environment_instance = None
        self.random_state = None  # type: np.random.RandomState
        self._cutoff_time = None
        self._hyper_parameters = None

        """
        Whether ot not to display a video of the agent execution at
        each episode
        """
        self.display_video = True

        """
        list that contains the amount of time-steps of the episode.
        It is used as a way to score the performance of the agent.
        """
        self.timesteps_of_episode = None

        """
        List that contains the amount of reward given to the agent in
        each episode
        """
        self.reward_of_episode = None

        # Dictionary of Q-values
        self.q = {}

        # Default hyper-parameters for Q-learning
        self._alpha = 0.5
        self._gamma = 0.9
        self._epsilon = 0.1

        # Amount of episodes to run for each run of the agent
        self.episodes_to_run = 3000
        self.actions = None

        """
        Adding hyper-parameter for softmax policy
        (named 'temperature parameter')
        """
        self._tau = 25

        """
        Matrix with 3 columns, where each row represents the action,
        reward and next state obtained from the agent executing an action
        in the previous state
        """
        self.action_reward_state_trace = []

    def set_hyper_parameters(self, hyper_parameters):
        """
        Method that passes the hyper_parameter configuration vector
        to the RL agent.
            :param hyper_parameters: A list containing the hyper-parameters
                                     that are to be set in the RL algorithm.
        """
        self._hyper_parameters = hyper_parameters

        for key, value in six.iteritems(hyper_parameters):
            if key == 'alpha':  # Learning-rate
                self._alpha = value

            if key == 'gamma':
                self._gamma = value

            if key == 'epsilon':
                self._epsilon = value

            if key == 'tau':
                self._tau = value

    def set_cutoff_time(self, cutoff_time):
        """
        Method that sets a maximum number of time-steps for each agent episode.
        :param cutoff_time:
        """
        self._cutoff_time = cutoff_time

    def init_agent(self, is_slippery=False):
        """
        Initializes the reinforcement learning agent with a
        default configuration.
        """
        if is_slippery:
            self._environment_instance = gym.make('FrozenLake-v0')
        else:
            """
            A Frozen Lake environment is registered with Slippery turned as
            False so it is deterministic
            """
            register(id='FrozenLakeNotSlippery-v0',
                     entry_point='gym.envs.toy_text:FrozenLakeEnv',
                     kwargs={'map_name': '4x4', 'is_slippery': False},
                     max_episode_steps=100,
                     reward_threshold=0.78)

            self._environment_instance = gym.make('FrozenLakeNotSlippery-v0')

        self.actions = range(self._environment_instance.action_space.n)

        # environment is seeded
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        if self.display_video:
            # video_callable=lambda count: count % 10 == 0)
            self._environment_instance = gym.wrappers.Monitor(
                                            self._environment_instance,
                                            '/tmp/frozenlake-experiment-1',
                                            force=True
                                        )


    def restart_agent_learning(self):
        """
        Restarts the reinforcement learning agent so it starts learning
        from scratch, in order to avoid bias with previous learning experience.
        """
        # last run is cleared
        self.timesteps_of_episode = []
        self.reward_of_episode = []
        self.action_reward_state_trace = []

        # q values are restarted
        self.q = {}

    def destroy_agent(self):
        """
        Destroys the reinforcement learning agent, in order to
        instantly release the memory the agent was using.
        """
        self._environment_instance.close()

    def learn_q_learning(self, state, action, reward, next_state):
        """
        Performs a Q-learning update for a given state transition

        Q-learning update:
        Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        new_max_q = max([self.q.get((next_state, a), 0.0)
                         for a in self.actions])
        old_value = self.q.get((state, action), 0.0)

        self.q[(state, action)] = old_value + self._alpha * (reward + self._gamma * new_max_q - old_value)

    def learn_sarsa(self, state, action, reward, next_state, next_action):
        """
        Performs a SARSA update for a given state transition:

        SARSA update:
                Q(s, a) = Q(s, a) + alpha *
                                  (reward(s,a) + (gamma * Q(s', a')) - Q(s,a))
        """
        new_q = self.q.get((next_state, next_action), 0.0)
        old_value = self.q.get((state, action), 0.0)

        self.q[(state, action)] = old_value + self._alpha * (reward + self._gamma * new_q - old_value)

    def choose_action(self, state, policy):
        """
        Chooses an action according to the learning previously performed
        """
        if policy == 'epsilon-greedy':
            q = [self.q.get((state, a), 0) for a in self.actions]
            max_q = max(q)

            if self.random_state.uniform() < self._epsilon:
                # A random action is returned
                return self.random_state.choice(self.actions)

            count = q.count(max_q)

            # In case there're several state-action max values
            # we select a random one among them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == max_q]
                i = self.random_state.choice(best)
            else:
                i = q.index(max_q)

            action = self.actions[i]

            return action

        elif policy == 'softmax':
            q = np.array([self.q.get((state, a), 0) for a in self.actions])
            # Q(s, a) / tau
            q_tau = q / self._tau  # Vector

            exp_q = np.exp(q_tau)   # e^(Q(s, a) / tau)
            exp_q_sum = sum(exp_q)

            exp_q_div = exp_q / exp_q_sum

            # Random floating point number in [0, 1)
            x = self.random_state.uniform()

            # Probabilidades acumuladas
            for idx, cum_prob in enumerate(np.cumsum(exp_q_div)):
                if x < cum_prob:
                    return self.actions[idx]

        else:
            print('Not valid policy')

    def run(self, algorithm='q-learning', policy='epsilon-greedy', penalty=False):
        """
        Runs the reinforcement learning agent with a given configuration.
        """
        for i_episode in range(self.episodes_to_run):
            """
            An instance of an episode is run until it fails or until it
            reaches 200 time-steps
            """

            # Resets the environment, obtaining the first state observation
            observation = self._environment_instance.reset()

            # A number of four digits representing the actual state is obtained
            state = observation

            if algorithm == 'sarsa':
                # Choose A from S using policy derived from Q
                # Epsilon-Greedy Policy
                action = self.choose_action(state, 'epsilon-greedy')

            for t in range(self._cutoff_time):

                if algorithm == 'q-learning':
                    # Pick an action based on the current state
                    action = self.choose_action(state, policy)

                # Execute the action and get feedback
                observation, reward, done, info = self._environment_instance.step(action)

                # current state transition is saved
                self.action_reward_state_trace.append(
                    [action, reward, observation]
                )

                # Digitize the observation to get a state
                next_state = observation

                if algorithm == 'sarsa':
                    next_action = self.choose_action(
                                    next_state, 'epsilon-greedy'
                                )

                if not done:
                    if algorithm == 'q-learning':
                        self.learn_q_learning(
                            state, action, reward, next_state
                        )
                    elif algorithm == 'sarsa':
                        self.learn_sarsa(
                            state, action, reward, next_state, next_action
                        )
                        action = next_action

                    state = next_state
                else:
                    # Episode finished because the agent fell into a hole
                    if reward == 0:
                        """
                        The default reward can be overrided by a hand-made
                        reward (below) for example to punish the agent
                        for falling into a hole
                        """
                        if penalty:
                            reward = -1
                        else:
                            # Replace this number to override the reward
                            reward = 0

                    if algorithm == 'q-learning':
                        self.learn_q_learning(
                            state, action, reward, next_state
                        )
                    elif algorithm == 'sarsa':
                        self.learn_sarsa(
                            state, action, reward, next_state, next_action
                        )

                    self.timesteps_of_episode = np.append(
                                                    self.timesteps_of_episode,
                                                    [int(t + 1)]
                                                )
                    self.reward_of_episode = np.append(
                                                self.reward_of_episode, reward
                                            )
                    break

        return self.reward_of_episode.mean()
