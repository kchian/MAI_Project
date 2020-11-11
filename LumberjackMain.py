try:
    from malmo import MalmoPython
except:
    import MalmoPython
import os
import sys
import time
import json
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from LumberjackEnvironment import getXML
from LumberjackQNet import QNetwork

#Hyperparameters
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
START_TRAINING = 500
LEARN_FREQUENCY = 1
LEARNING_RATE = 1e-4

SIZE = 10 #Dimensions of map
PATH = r"Models\state_dict_model%d.pt" #Path to save model
COLOURS = {'wood': (162, 0, 93), 'leaves':(162, 232, 70), 'grass':(139, 46, 70)}

ACTION_DICT = {
    0: 'move 1',  # Move one block forward
    1: 'turn 1',  # Turn 90 degrees to the right
    2: 'turn -1',  # Turn 90 degrees to the left
    3: 'attack 1',  # Destroy block
    # 4: 'pitch 1',
    # 5: 'pitch -1',
    # 6: 'move 0',
    # 7: 'turn 0',
    # 8: 'attack 0',
    # 9: 'pitch 0'
}

def init_malmo(agent_host):
    #Record Mission 
    my_mission = MalmoPython.MissionSpec(getXML(MAX_EPISODE_STEPS, SIZE), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recording' + str(int(time.time())) + '.tgz']))
    my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)

    my_mission.requestVideoWithDepth(800, 500)
    my_mission.setViewpoint(0)

    # Attempt to start a mission:
    max_retries = 1
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)
                continue
    return agent_host

def get_observation(world_state):
    obs = np.zeros((4, 800, 500))

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if len(world_state.video_frames):
            if world_state.video_frames[0].channels == 4:
                pixels = world_state.video_frames[0].pixels
                obs = np.reshape(pixels, (4, 800, 500))
                break
            else:
                # If a lot of these show up, we might want something to find the first frame with depth
                print('no depth found')
    return obs

def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    print(obs, action, next_obs, reward, done)
    return obs, action, next_obs, reward, done
  
def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()

def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Diamond Collector')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 

def get_action(obs, q_network, epsilon):
    """
    Select action according to e-greedy policy

    Args:
        obs (np-array): current observation, size (obs_size)
        q_network (QNetwork): Q-Network
        epsilon (float): probability of choosing a random action

    Returns:
        action (int): chosen action [0, action_size)
    """
    #------------------------------------
    #
    #   TODO: Implement e-greedy policy
    #
    #-------------------------------------

    # Prevent computation graph from being calculated
    with torch.no_grad():
        # Calculate Q-values fot each action
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        if random.random() < epsilon:
            action_idx = randint(len(ACTION_DICT))
        else:
        # Select action with highest Q-value
            action_idx = torch.argmax(action_values).item()
        
    return action_idx

def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((4, 800, 500), len(ACTION_DICT))
    target_network = QNetwork((4, 800, 500), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE, weight_decay = 0)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs = get_observation(world_state)

        # Run episode
        print("\nRunning")
        while world_state.is_mission_running:
            # Get action
            action_idx = get_action(obs, q_network, epsilon)
            command = ACTION_DICT[action_idx]

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(.1)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS:
                done = True
                time.sleep(2)
                '''
                    or \
                (obs[0, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1 and \
                obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 0 and \
                command == 'move 1'):
                '''

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state) 

            # Get reward
            #TODO Add reward for moving closer to tree
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 100 == 0:
            #ADD save
            torch.save(q_network.state_dict(), PATH%num_episode)
            log_returns(steps, returns)
            print()


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)