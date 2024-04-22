import sys

import os
import pickle

import torch.multiprocessing as mp

def run_episode(env, agents, render_mode=None, render_file=None):
    """
    Runs 1 episode in env with any combination of Agents.
    To do:
        - right now, we ignore negative reward at the end, generalize this to accomodate any pettingzoo fully cooperative environment
        - right now, we assume no agents reaches terminal state in the first round because only 2 players, fix this to accomodate more players

    Preconditions:
        - env needs to be pettingzoo.classic.hanabi
        - len(agents) == env.num_agents
        - env.render_mode == render_mode
        - agents.select_action() only returns legal actions
    
    Parameters:
        env : pettingzoo environment
        agents : list of Agent
        render_mode : eg. for pettingzoo.classic.hanabi.hanabi.env, None or "human"
        render_file : file path for the output
    
    Returns:
        float: episode return

    Side Effects:
        - if render_mode == "human", stdout is changed to render_file during execution
    """
    if render_file is not None:
        # create file if it doesn't exist
        f = open(render_file, 'w')
        # redirect stdout to file (because env.render() prints directly to stdout)
        sys.stdout = f

    # assign agents to players
    agent = {}
    for i in range(env.num_agents):
        agent[env.possible_agents[i]] = agents[i]

    # for print purpose only
    first_player = env.agent_selection
    last_player = get_previous(env.agent_selection, env.possible_agents)

    # keeping track of things
    score = {} # all agents keep track of highest achieved score, should all be the same at the end but just in case for future use
    for player in env.possible_agents:
        score[player] = 0
    final_score = 0 # exactly the same as above, just for symmetry OCPD
    s = {} # all agents keep track of the last received state
    a = {} # all agents keep track of the last action taken
    r = {} # the last received reward
    terminated = {} # all agents keep track of termination status
    # technically should do the same for truncated but not needed for hanabi
    turn = 0 # keep track of the number of turns within an episode
    game_terminated = False # terminated['player_0'] or terminated['player_1'] or ...

    # initializing s, a for for all agents
    # start keeping track of score[player]
    for player in env.agent_iter(env.num_agents):
        obs, r[player], terminated[player], truncated, info = env.last()

        if r[player] > 0:
            score[player] += r[player]
        if player == first_player and render_mode == "human":
            print(f"[TURN {turn} : {player}] ************************")
            print(f"------ receives INITIAL STATE ------")
            env.render()
        if terminated[player] or truncated and render_mode == "human":
            print()
            print(f"GAME TERMINATED at turn {turn} with {score[player]}.")
            # Function should return and not go into next loop

        # choosing action
        action = agent[player].select_action(obs)
        agent[player].increment_step()
        
        # save state and action for the next iteration
        s[player] = obs
        a[player] = action
        
        if render_mode == "human":
            print()
            print(f"------ takes ACTION ------")
            print(f"action = {action}")
            print()
            print(f"[TURN {turn+1} : {get_next(player, env.possible_agents)}] ************************")
            if player != last_player:
                print(f"------ receives INITIAL STATE ------")
            else:
                print(f"------ receives NEXT STATE ------")
        env.step(action)
        turn += 1

    # main episode loop
    for player in env.agent_iter(): # technically should make sure first agent == env.agent_selection
        
        # looks like not needed to break out of loop, step() and agent_iter() will take care of that
        """
        all_terminated = all(terminated.values())
        print(f"all_terminated: {all_terminated}")
        if all_terminated:
            # print rewards for all agents
            if render_mode == "human" :
                print()
                for player in env.possible_agents:
                    print(f"{player} receives r={r[player]}")
            break
        """
        obs, r[player], terminated[player], truncated, info = env.last()
        if r[player] > 0:
            score[player] += r[player]

        if render_mode == "human" and not terminated[player] and not truncated:
            print()
            print(f"------ receives REWARD ------")
            print(f"r = {r[player]}")

        # choosing action
        if not terminated[player] and not truncated:
            action = agent[player].select_action(obs)
            agent[player].increment_step()
        else:
            action = None # agent should stop stepping
            obs = None # obs is the next_state in the transition
            if r[player] < 0:
                r[player] = 0 # reward shaping: not penalizing for not reaching perfect score

            if not game_terminated:
                if render_mode == "human":
                    print()
                    print(f"GAME TERMINATED at turn {turn} with score {int(score[player])}.")
                game_terminated = True
                final_score = score[player]

        # Store the transition in memory
        agent[player].store_transition(s[player], a[player], obs, r[player])
        s[player] = obs
        a[player] = action

        # OFFLINE LEARNING            
        # # Perform one step of the optimization (on the policy network)
        # agent[player].optimize_model()
        # # Soft update of the target network's weights
        # agent[player].update_target()

        if render_mode == "human" and not game_terminated:
            print()
            print(f"------ takes ACTION ------")
            print(f"action = {action}")
            print()
            print(f"[TURN {turn+1} : {get_next(player, env.possible_agents)}] ************************")
            print(f"------ receives NEXT STATE ------")
        env.step(action) # env.render() being called
        turn += 1

    # print the last reward for all agents
    if render_mode == "human":
        for player in env.possible_agents:
            print(f"{player}'s last reward r = {int(r[player])}")

    if render_file is not None:
        # close file and restore stdout
        sys.stdout = sys.__stdout__
        f.close()

    # return score
    return final_score

# Helper methods for visualizing the game
def get_next(element, set_list):
    try:
        index = set_list.index(element)
        next_index = (index + 1) % len(set_list)  # Wrap around if we reach the end of the list
        return set_list[next_index]
    except ValueError:
        return f"Element {element} not found in the list"

def get_previous(element, set_list):
    try:
        index = set_list.index(element)
        previous_index = (index - 1) % len(set_list)  # Wrap around if we reach the end of the list
        return set_list[previous_index]
    except ValueError:
        return f"Element {element} not found in the list"

def run_trial(env_constructor, agents, n_episodes, trial_output=[], returns_output_episodes=[], agents_output_episodes=[], output_dir=os.getcwd(), data_id='', verbose=True, trial_number=''):
    """
    Initializes the environment within the function for encapsulation.
    Creates a copy of agents before running the trial for encapsulation.
    Runs 1 trial of the env with n_episodes.
    
    Preconditions:
    - all agents have steps_done attribute, optimize_model() and update_target() methods

    Input:
    - env: requires env.render_mode == None
    - agents: list of agents
    - trial_returns: starting list of episode returns
    - delta_episodes: how many more episodes to run
    - results_episode_list: list of episode index to save results
    - output_dir: directory to save the model
    - data_id: id for the trial settings
    - verbose: print scores only to console
    - trial_number: for print purpose only
    
    Output:
    - trial_output: list of episode returns
    """
    
    env = env_constructor(render_mode=None, players=len(agents))
    agents = [agent.copy() for agent in agents]
    total_episodes = len(trial_output)

    for _ in range(n_episodes):
        env.reset()
        
        score = run_episode(env, agents, render_mode=None)
        trial_output.append(score)
        
        # OFFLINE LEARNING
        for agent in agents:
            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()
            # Soft update of the target network's weights
            agent.update_target()

        if total_episodes in agents_output_episodes:
            # save agents
            for player, agent in enumerate(agents):
                filename = os.path.join(output_dir, f"{data_id}__ep{total_episodes}__player{player}.pt")
                agent.save(filename)
        if total_episodes in returns_output_episodes:
            # save trial_returns
            filename = os.path.join(output_dir, f"{data_id}__ep{total_episodes}__trial_output.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(trial_output, f)
        
        if verbose:
            if trial_number == '':
                prefix = ''
            else:
                prefix = f"Trial {trial_number} "
            print(f"{prefix}Episode {total_episodes}  score: {int(score)}")
        total_episodes += 1

    return trial_output

def run_trials_sequential(env_constructor, agents, num_trials, num_episodes, returns_output_episodes, agents_output_episodes, output_dir, data_id, verbose=True):
    trials_returns = []
    for trial in range(num_trials):
        trials_returns.append(run_trial(env_constructor, agents, num_episodes, [], returns_output_episodes, agents_output_episodes, output_dir, f"{data_id}__trial{trial}", verbose, trial))
        if verbose:
            print()
    return trials_returns

# Helper method for parallelizing the trials
# args is a tuple of all the arguments needed for run_trial + process id
def run_trial_wrapper(args):
    return run_trial(*args)

# Version 1: using mp.Process
# import multiprocessing as mp
# def run_trials_parallel(env_constructor, agents, num_trials, num_episodes, returns_output_episodes, agents_output_episodes, output_dir, data_id, verbose=True):
#     num_processes = mp.cpu_count()
#     input_list = []
#     for trial in range(num_trials):
#         input_list.append((env_constructor, agents, num_episodes, [], returns_output_episodes, agents_output_episodes, output_dir, f"{data_id}__trial{trial}", False, trial))
#
#     processes = []
#     trials_returns = []
#
#     for rank in range(num_trials):
#         process = mp.Process(target=run_trial_wrapper, args=(input_list[rank],))
#         process.start()
#         print(f"Started process/trial {rank}")
#         processes.append(process)
#
#     for process in processes:
#         result = process.join()
#         trials_returns.append(result)
#
#     print("All processes/trials finished")
#     return trials_returns

# Version 2: using mp.Pool
def run_trials_parallel(env_constructor, agents, num_trials, num_episodes, returns_output_episodes, agents_output_episodes, output_dir, data_id, verbose=True):
    input_list = []
    for trial in range(num_trials):
        input_list.append((env_constructor, agents, num_episodes, [], returns_output_episodes, agents_output_episodes, output_dir, f"{data_id}__trial{trial}", False, trial))

    num_processes = min(mp.cpu_count(), num_trials)
    with mp.get_context("spawn").Pool(processes=num_processes) as pool:
        trials_returns = pool.map(run_trial_wrapper, input_list)
    return trials_returns