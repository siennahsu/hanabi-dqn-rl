# hanabi-dqn-rl
Using reinforcement learning algorithms Deep Q-Network (DQN), Double DQN, and Prioritized Experience Replay to solve the game of Hanabi.

This is in collaboration with Ronald Truong.

The project repo contains six .py files and one .ipynb file. Their uses are as such:

- main.ipynb contains the code to tune and run the experiments fully.
- network_models.py contains the neural network class we use for our agents.
- agents.py contains all the agent classes.
- memory_models.py contains the classes used for experience replay.
- training_offline.py contains the code to run an offline trial.
- training_online.py contains the code to run an online trial.
- testing.py is used for purely playing the game and not training. 

The file report.pdf discusses the project in depth.

To run our code, the packages below need to be installed:

numpy
gymnasium
pettingzoo
shimmy[openspiel] 
pytorch
torchrl
