import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import os
import ale_py



class DQNBreakout(gym.Wrapper):

    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
        super(DQNBreakout, self).__init__(env)

        self.repeat = repeat
        self.lives = env.unwrapped.ale.lives()
        self.frame_buffer = []
        self.device = device
        self.image_shape = (84, 84) # could be larger, smaller but is
        # dimensions in the paper

    # common gymnasium function
    def step(self, action):
        total_reward = 0
        done = False

        for i in range(self.repeat):
            observation, reward, done, truncated, info = self.env.step(action)

            total_reward += reward

            # print(info, total_reward)

            current_lives = info['lives']

            if current_lives < self.lives:
                total_reward = total_reward - 1 # arbitrary, could be a different 'punishment'
                self.lives = current_lives # decrementing the lives

            # print(f"Lives: {self.lives} Total Reward: {total_reward}")

            self.frame_buffer.append(observation)  # state and observation is
            # used interchangeably in this code

            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)

        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float() # what are we doing here with the view function?
        # It adds a dimension right?
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1).float()
        done = done.to(self.device)

        return max_frame, total_reward, done, info

    # overriding the reset (common gymnasium function)
    def reset(self):
        self.frame_buffer = [] # clearing the frame buffer (which makes sense)

        observation, _ = self.env.reset() # functionally same as env.reset()[0], what is this function
        # and why do I see it so much? Just to get the initial state/observation?

        self.lives = self.env.unwrapped.ale.lives() # resets lives to 5 -- could I just set it to five manually?

        observation = self.process_observation(observation)

        return observation

    def process_observation(self, observation):

        # TODO add content

        img = Image.fromarray(observation)
        # Standardization and reducing dimensionality

        img = img.resize(self.image_shape)

        img = img.convert("L") # converts to grayscale

        img = np.array(img) # why this instead of ndarry?
        img = torch.from_numpy(img)
        img = img.unsqueeze(0) # adds dimension to first axis
        img = img.unsqueeze(0) # adds another dimension to first axis
        # why are we adding two dimensions to the image again (15:30 Part2)?

        img = img / 255.0

        img = img.to(self.device)

        return img