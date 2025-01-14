import copy

import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from plot import LivePlot
import numpy as np
import time


class ReplayMemory:
    """functions as the agent's memory"""
    def __init__(self,
                 capacity,
                 device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    def insert(self, transition):
        """transition is the data unit at play here
        transition a tuple of (state, action, reward, done, next_state),
        """
        transition = [item.to('cpu') for item in transition] # this allows for having less gpu memory
        # pushes the 'memory' to the cpu to store and then, when we 'retrieve' those memories
        # we put those tensors back on the gpu to avoid running out of gpu memory

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch) # go over this again? Asterisks always get me

        return [torch.cat(items).to(self.device) for items in batch] # converting back to the gpu at the time of sampling

    def can_sample(self, batch_size):
        """makes sure you have enough """
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        """overriding length method"""
        return len(self.memory) #

class Agent:
    """Agent class"""

    def __init__(self,
                 model,
                 device='cpu',
                 epsilon=1.0,
                 min_epsilon=0.1,
                 nb_warmup=10000,  # the period over which epsilon should decay
                 nb_actions=None,
                 memory_capacity=10000,
                 batch_size=32,
                 learning_rate=0.00025):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2) # what does this do? Get chat to explain the reasoning here
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate) # I guess model.parameters() is just
        # the expected way to pass your 'model' to the optimizer function in pyTorch?

        print(f"Starting epsilon is {self.epsilon}")
        print(f"Epsilon decay is {self.epsilon_decay}")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1)) # pretty sure this is just an integer between 0 and 3 but maybe check?
        else:
            av = self.model(state).detach() # what does detach do?

            # [0.11, 0.45, 0.22, 0.3] example and np.argmax  returns
            # 0.45 and just gets the highest number, resulting in the
            # output with the highest confidence
            return torch.argmax(av, dim=1, keepdim=True)

    def train(self, env, epochs):
        stats = {'Returns': [], "AvgReturns": [], "EpsilonCheckpoint": []}

        plotter = LivePlot()

        for epoch in range(1, epochs+1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, info = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size) # giving a batch size

                    qsa_b = self.model(state_b).gather(1, action_b) # q state action for the batch
                    # what does this do? what does gather do?
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

                    # honestly not sure if this works but it appears so? Come back
                    # to this if you encounter problems and try the operation commented out

                    # If done_b is a boolean tensor
                    # done_b_float = done_b.float()  # Convert to float: True->1.0, False->0.0
                    # not_done_b = 1.0 - done_b_float

                    # Compute target_b
                    # target_b = reward_b + not_done_b * self.gamma * next_qsa_b
                    not_done_b = torch.logical_not(done_b)
                    not_done_b = not_done_b.float()
                    # target_b = reward_b + ~done_b * self.gamma * next_qsa_b #what does the tilde do?
                    target_b = reward_b + not_done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad() # zeros out the gradient

                    loss.backward() # actual backprop step
                    self.optimizer.step()

                state = next_state

                ep_return += reward.item()

            stats["Returns"].append(ep_return)

            # doing epsilon decay
            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            # periodically checking the model and saving
            if epoch % 10 == 0:
                self.model.save_the_model()
                print(" ")

                average_returns = np.mean(stats["Returns"][-100:])

                stats["AvgReturns"].append(average_returns)
                stats["EpsilonCheckpoint"].append(self.epsilon)

                # prints the data episodically for the first 100 iterations and then, after 100
                # prints the average
                if (len(stats["Returns"])) > 100:
                    print(f"Epoch: {epoch} - Average return: {np.mean(stats["Returns"][-100:])} - Epsilon: {self.epsilon}")
                else:
                    print(
                        f"Epoch: {epoch} - Episode return: {np.mean(stats["Returns"][-1:])} - Epsilon: {self.epsilon}")

            if epoch  % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict()) # not sure what this is doing?
                # also, what is the target model? Not sure what that is?
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")

        return stats

    def test(self, env):
        for epoch in range(1, 3): # arbitrary number of times, can be modified
            state = env.reset()

            done = False

            for _ in range(1000):
                time.sleep(0.01) # helps it run a little closer to human readability
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break