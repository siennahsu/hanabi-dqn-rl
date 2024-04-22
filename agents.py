import random
import numpy as np
import torch
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

####################################################################
# TO DO: 
# - option to move tensors to device should be in main code
# - unify Agent.load() for super class DQNAgent
# - make save/load more minimal/efficient
# - currently saving the state of the agent, it works only if reloaded in the same machine and same python environment?
####################################################################

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, policy_net, target_net, optimizer, memory, batch_size, gamma, tau, eps_start, eps_end, eps_decay, steps_done=0):
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer
        self.memory = memory

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.steps_done = steps_done

    """
    To do:
    - reimplement to work for any number of players
    """
    def select_action(self, obs):
        mask = obs['action_mask']
        # convert the observation to a tensor
        obs = torch.tensor(obs['observation'], dtype=torch.float32, device=device).unsqueeze(0)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self.steps_done / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                actions = self.policy_net(obs) 

                min_action_value = torch.min(actions).item()

                for i in range(len(actions[0])):
                    if mask[i] == 0:
                        actions[0][i] = min_action_value - 1

                return actions.max(1).indices.view(1, 1).item()
        else:
            # calulate the probability of choosing each legal action randomly
            sum_of_ones = sum(mask)
            if sum_of_ones == 0:
                # return torch.tensor([random.randint(0, 19)], device=device, dtype=torch.long)
                print("All actions are illegal.")

            mask = mask / sum_of_ones
            action = np.random.choice([i for i in range(20)], p=mask)
            return torch.tensor([[action]], device=device, dtype=torch.long).item()
    
    def increment_step(self):
        self.steps_done += 1

    def store_transition(self, state, action, next_state, reward):
        state = torch.tensor(state['observation'], dtype=torch.float32, device=device).unsqueeze(0)
        if next_state is not None:
            next_state = torch.tensor(next_state['observation'], dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([[action]], device=device)
        reward = torch.tensor([int(reward)], device=device)
        
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions, info = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # update the priorities of the transitions in the case of prioritized replay buffer
        self.memory.update_priority(transitions, info, state_action_values.squeeze(1), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def copy(self):
        policy_net = self.policy_net.copy()
        target_net = self.target_net.copy()
        optimizer = self.optimizer.__class__(policy_net.parameters())
        optimizer.load_state_dict(self.optimizer.state_dict())
        return DQNAgent(policy_net, target_net, optimizer, self.memory.copy(), self.batch_size, self.gamma, self.tau, self.eps_start, self.eps_end, self.eps_decay, self.steps_done)

    def save(self, filename):
        torch.save({
            'agent': self,
        }, filename)

    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename)
        agent = checkpoint['agent']
        return agent

class DDQNAgent(DQNAgent):
    def __init__(self, policy_net, target_net, optimizer, memory, batch_size, gamma, tau, eps_start, eps_end, eps_decay, steps_done=0):
        super().__init__(policy_net, target_net, optimizer, memory, batch_size, gamma, tau, eps_start, eps_end, eps_decay, steps_done)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions, info = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list)
            
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        
        if len(non_final_next_states_list) > 0:
            with torch.no_grad():
    
                # obtain the optimal actions of s_t+1 using target net
                policy_net_optimal_action_batch = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
    
                # calculate the state action values of s_t+1 with respect to the target net's optimal action batch
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, policy_net_optimal_action_batch).squeeze(1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # update the priorities of the transitions in the case of prioritized replay buffer
        self.memory.update_priority(transitions, info, state_action_values.squeeze(1), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def copy(self):
        policy_net = self.policy_net.copy()
        target_net = self.target_net.copy()
        optimizer = self.optimizer.__class__(policy_net.parameters())
        optimizer.load_state_dict(self.optimizer.state_dict())
        return DDQNAgent(policy_net, target_net, optimizer, self.memory.copy(), self.batch_size, self.gamma, self.tau, self.eps_start, self.eps_end, self.eps_decay, self.steps_done)

class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, obs):
        mask = obs['action_mask']
        n_actions = len(mask)
        legal_actions = [i for i in range(n_actions) if mask[i] == 1]
        action = np.random.choice(legal_actions)
        return action

    def increment_step(self):
        pass

    def store_transition(self, state, action, next_state, reward):
        pass

    def optimize_model(self):
        pass

    def update_target(self):
        pass

    def copy(self):
        return RandomAgent()

    def save(self, filename):
        pass

class HumanAgent:
    def __init__(self, name):
        self.name = name

    def select_action(self, obs):
        mask = obs['action_mask']
        action = int(input(f"ENTER {self.name}'s action:"))
        # check if the action is valid
        while mask[action] == 0:
            print("Invalid action! Please try again.")
            action = int(input(f"ENTER {self.name}'s action:"))
        
        return action

    def increment_step(self):
        pass

    def store_transition(self, state, action, next_state, reward):
        pass

    def optimize_model(self):
        pass

    def update_target(self):
        pass

    def copy(self):
        return HumanAgent(self.name)

    def save(self, filename):
        pass