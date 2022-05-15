import gym
import numpy as np
import time

class QLearningAgent(object):
    def __init__(self, num_buckets = 10, num_episodes = 1000, lr = 0.1, discount = 1., min_expore = 0.05, decay = 50):
        self.num_buckets = num_buckets
        self.num_episodes = num_episodes
        self.lr = lr
        self.discount = discount
        self.min_expore = min_expore
        self.decay = decay
        
        self.env = gym.make('MountainCar-v0')
        self.explore_rate = self.get_explore_rate(0)
        self.Q_table = np.zeros((self.num_buckets, self.num_buckets) + (self.env.action_space.n, ))

    def discretize_state(self, obs):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high

        scaling = (env_high - env_low) / self.num_buckets
        position_bin = int((obs[0] - env_low[0]) / scaling[0])
        velocity_bin = int((obs[1] - env_low[1]) / scaling[1])

        return (position_bin, velocity_bin)

    def get_explore_rate(self, e):
        return max(self.min_expore, np.exp(-e / self.decay))

    def choose_action(self, state):
        if np.random.random() < self.explore_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def update_q_table(self, state, action, reward, new_state):
        self.Q_table[state][action] += self.lr * (reward + self.discount * np.max(self.Q_table[new_state]) - self.Q_table[state][action]) 

    def train(self):
        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())
            self.explore_rate = self.get_explore_rate(e)

            done = False
            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                self.update_q_table(current_state, action, reward, new_state)
                current_state = new_state

    def run(self):
        while True:
            current_state = self.discretize_state(self.env.reset())

            done = False
            while not done:
                self.env.render()
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                current_state = new_state
                time.sleep(.01)

agent = QLearningAgent()
agent.train()
agent.run()
