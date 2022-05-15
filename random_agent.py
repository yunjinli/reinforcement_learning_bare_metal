import gym

class RandomAgent(object):
    def __init__(self):
        self.env = gym.make('MountainCar-v0')

    def choose_action(self, state):
        return self.env.action_space.sample()

    def run(self):
        while True:
            current_state = self.env.reset()
            done = False
            
            while not done:
                self.env.render()
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                current_state = new_state

if __name__ == '__main__':
    agent = RandomAgent()
    agent.run()