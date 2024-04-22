import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_layers):
        """
        Design by Contract: len(hidden_layers) > 0 

        n_observations: dimension of state space
        n_actions: size of action space
        hidden_layers: list of number of neurons in each hidden layer
        """

        # inherit the nn.Module class
        super(DQN, self).__init__()

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_layers = hidden_layers

        # Create input layer
        self.layers = nn.ModuleList([nn.Linear(n_observations, hidden_layers[0])])

        # Create hidden layers
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

        # Create output layer
        self.layers.append(nn.Linear(hidden_layers[-1], n_actions))


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
    def reset(self):
        # Clear existing layers
        self.layers = nn.ModuleList()

        # Create input layer
        self.layers.append(nn.Linear(self.n_observations, self.hidden_layers[0]))

        # Create hidden layers
        for i in range(len(self.hidden_layers)-1):
            self.layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))

        # Create output layer
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.n_actions))

    def copy(self):
        network = DQN(self.n_observations, self.n_actions, self.hidden_layers)
        network.load_state_dict(self.state_dict())
        network.eval()
        return network