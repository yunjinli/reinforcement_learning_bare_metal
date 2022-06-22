from network import MLP
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np

class PPO:
    def __init__(self, env):
        # First, we need to know in which environemnt we are working on
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        self._init_hyperparameters()
        
        ## Algorithm Step 1 
        ## Initialize actor and critic network
        self.actor = MLP(self.obs_dim, self.act_dim)
        self.critic = MLP(self.obs_dim, 1)

        # Create our variable for the matrix.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        ## Initialize the actor optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        
        ## Initialize the critic optimizer
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)


    def learn(self, total_time_steps):
        t = 0 # Time steps which we have generated so far
        
        while t < total_time_steps:
            ## Algorithm Step 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            V, _ = self.evaluate(batch_obs, batch_acts)

            ## Algorithm Step 5
            ## Calculate advantage
            A_k = batch_rtgs - V.detach()
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                ## The actor loss according to the pseudocode is
                actor_loss = (-torch.min(surr1, surr2)).mean() ## We take mean() here because we have summation and 1/N in the equation
                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()


                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

            # Calculate how many timesteps we collected this batch   
            t += np.sum(batch_lens)

    def _init_hyperparameters(self):
        self.time_steps_per_batch = 4800
        self.max_time_steps_per_episode = 1600
        self.gamma = 0.9
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def rollout(self):
        ## Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []            # batch rewards
        batch_rewards_to_go = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        t = 0

        while t < self.time_steps_per_batch: ## Algorithm Step 2

            ## Reward sequence in "THIS" episode
            ep_rewards = []

            obs = self.env.reset()
            done = False

            for ep in range(self.max_time_steps_per_episode):
                ## Increment timesteps so far
                t += 1

                ## Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                # self.env.render()
                ## Collect reward, action, and log probability
                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done: ## Either the episode ends or finishes, we break the for loop
                    break

            ## Collect episodic length and rewards
            batch_lens.append(ep + 1) ## ep start from 0
            batch_rewards.append(ep_rewards)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # Algorithm Step 4
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens

    def get_action(self, obs):
        ## First, query the actor network for a mean action
        mean = self.actor(obs)

        ## Create the Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat) ## consider it as a normal distribution in high dimensional space

        ## Sample an action from the distribition and get its log-probability
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []

        ## Note that we calculate the reward-to-go typically from the last state
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0 ## This accumulative discounted reward

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0,  discounted_reward) ## make sure the order is still from 1 to k not k to 1, so we always "INSERT" new discounted reward in the front

        ## Convert rewards-to-go into tensor
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype = torch.float)

        return batch_rewards_to_go

    def evaluate(self, batch_obs, batch_acts):
        ## Query the critic network for a value V for each observation in batch_obs
        V = self.critic(batch_obs).squeeze()
        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts) ## Note that we don't sample action here because we already did it in rollout()
        # Return predicted values V and log probs log_probs
        return V, log_probs


if __name__ == '__main__':
    import gym 
    env = gym.make('Pendulum-v1')
    model = PPO(env)
    model.learn(10000)