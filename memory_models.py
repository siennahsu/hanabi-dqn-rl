from collections import namedtuple, deque
import random
from torchrl.data import ListStorage, PrioritizedReplayBuffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return (random.sample(self.memory, batch_size), None)
    
    def update_priority(self, sample, info, state_action_values, expected_state_action_values):
        pass

    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory = deque([], maxlen=self.capacity)

    def copy(self):
        return ReplayMemory(self.capacity)

def identity(x):
    return x

class PrioritizedReplayMemory:

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        # self.memory = TensorDictPrioritizedReplayBuffer(alpha=alpha, beta=beta, storage=LazyTensorStorage(capacity), collate_fn=collate_fn)
        self.memory = PrioritizedReplayBuffer(alpha=alpha, beta=beta, storage=ListStorage(capacity), collate_fn=identity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

    def push(self, *args):
        """Save a transition"""
        self.memory.add(Transition(*args))

    def sample(self, batch_size):
        return self.memory.sample(batch_size, return_info=True)
      
    def update_priority(self, sample, info, state_action_values, expected_state_action_values):
        """both action_values have to be squeezed"""
        # proportional-based method
        priority = abs(expected_state_action_values-state_action_values)+0.00001 # 0.00001 is noise
        self.memory.update_priority(info["index"], priority)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.empty()

    def copy(self):
        return PrioritizedReplayMemory(self.capacity, self.alpha, self.beta)