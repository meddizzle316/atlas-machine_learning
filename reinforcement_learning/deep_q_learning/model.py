import os.path

import torch
import torch.nn as nn
import os

class AtariNet(nn.Module):

    def __init__(self, nb_actions=4):
        """
        nb_actions = number of possible actions
        """
        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        #splitting the neural net -- agent side
        self.action_value1 = nn.Linear(3136, 1024) # the part making the decisions
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_actions) # determines the action taken by the agent


        # critic side
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1) # telling whether a given state is valuable to the agent


    def forward(self, x):
        x = torch.Tensor(x)

        # running the base conv before splitting
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)


        # splitting the network -- critic side
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))


        # splitting the network -- action side
        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value3(action_value))

        # combining both action and state
        output = state_value + (action_value - action_value.mean()) # to avoid 'double representing' the action state value? Not sure
        # should come back to this
        return output

    def save_the_model(self, weights_file_name='models/latest.pt'):
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(self.state_dict(), weights_file_name) # I guess state_dict() is a built in Torch module property?


    def load_the_model(self, weights_file_name='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_file_name))
            print(f"Successfully loaded weights file {weights_file_name}")
        except:
            print(f"No weights file available at {weights_file_name}")
