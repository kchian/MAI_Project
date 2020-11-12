import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from numpy.random import randint
from LumberjackEnvironment import getXML
from LumberjackQNet import QNetwork

#Hyperparameters
MAX_EPISODE_STEPS = 30
MAX_GLOBAL_STEPS = 100000
REPLAY_BUFFER_SIZE = 10000
MIN_EPSILON = .1
BATCH_SIZE = 70
GAMMA = .9
TARGET_UPDATE = 100
START_TRAINING = 500
LEARN_FREQUENCY = 100
LEARNING_RATE = 1e-4
EPSILON_DECAY = .999**LEARN_FREQUENCY

SIZE = 10 #Dimensions of map
PATH = r"state_dict_model%d.pt" #Path to save model
LOAD = False
MODELNUM = 1000
MODEL = r"state_dict_model%d.pt"
COLOURS = {'wood': (162, 0, 93), 'leaves':(162, 232, 70), 'grass':(139, 46, 70)}

ACTION_DICT = {
    0: 'move .5',  # Move forward
    1: 'turn 0.25',  # Turn 22.5 degrees to the right
    2: 'turn -0.25',  # Turn 22.5 degrees to the left
    # 3: 'attack 1',  # Destroy block
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
    # my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recording' + str(int(time.time())) + '.tgz']))
    # my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)
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
            for frame in reversed(world_state.video_frames):
                if frame.channels == 4:
                    break
            if frame.channels == 4:
                pixels = world_state.video_frames[0].pixels
                obs = np.reshape(pixels, (4, 800, 500))
                break
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
    print("Batching", end="")
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    print(".", end = "")
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    print(".", end = "")
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    print(".", end = "")
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    print(".", end = "")
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    print(".", end = "")
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    #print(obs, action, next_obs, reward, done)
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
    plt.title('Reach the tree')
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
    # Prevent computation graph from being calculated
    with torch.no_grad():
        # Calculate Q-values fot each action
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        if random.random() < epsilon:
            print("r", end="")
            action_idx = randint(len(ACTION_DICT))
        else:
        # Select action with highest Q-value
            print("q", end="")
            action_idx = torch.argmax(action_values).item()
    return action_idx

def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    #Init networks
    q_network = QNetwork((4, 800, 500), len(ACTION_DICT))
    target_network = QNetwork((4, 800, 500), len(ACTION_DICT))
    if LOAD:
        q_network.load_state_dict(torch.load(PATH%MODELNUM))
        q_network.eval()
        target_network.load_state_dict(torch.load(PATH%(MODELNUM+10000)))
        target_network.eval()
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE, weight_decay = 0)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    last = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []
    
    if LOAD:
        global_step = 300
        epsilon = .8
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
        while world_state.is_mission_running:
            # Get action
            action_idx = get_action(obs, q_network, epsilon)
            command = ACTION_DICT[action_idx]

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(.3)

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
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            for o in world_state.observations:
                msg = o.text
                observations = json.loads(msg)
                reward -= observations['distanceFromTree']
            for f in world_state.video_frames:
                if f.frametype == MalmoPython.FrameType.COLOUR_MAP:
                    center_x = 400
                    center_y = 250
                    if (f.pixels[center_x*center_y], f.pixels[center_x*center_y*2], f.pixels[center_x*center_y*3]) == COLOURS['wood']:
                        reward += 10
            episode_return += reward
            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            last+=1

            #Sleep until death if falling
            if world_state.observations:
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                if observations['YPos']<-2:
                    obs, action_idx, next_obs, reward, done = replay_buffer[-1]
                    replay_buffer.append((obs, action_idx, next_obs, reward-10000, done))
                    episode_return-=10000
                    time.sleep(3)
                    break
        if global_step > START_TRAINING and last>=LEARN_FREQUENCY:
            print("Learning")
            last = 0
            batch = prepare_batch(replay_buffer)
            loss = learn(batch, optim, q_network, target_network)
            episode_loss += loss

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY

            if last>=TARGET_UPDATE:
                target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        #Save model and log returns every hundred episodes
        if num_episode%25==0:
            print("Saved Model")
            torch.save(q_network.state_dict(), PATH%num_episode)
            torch.save(target_network.state_dict(), PATH%(num_episode+10000))
            log_returns(steps, returns)


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