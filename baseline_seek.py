
try:
    from malmo import MalmoPython
except:
    import MalmoPython
from PigCatchSmall import getXML
from FrameProcessor import draw_helper
import os
import sys
import time
import json
import random
import matplotlib.pyplot as plt 
import numpy as np

WIDTH, HEIGHT = (20, 20)
pig_color = np.array([1, 57, 110])

def binary_conv_obs(obs):
    out = np.zeros((WIDTH, HEIGHT))
    for row in range(WIDTH):
        for col in range(HEIGHT):
            if (obs[row][col] == pig_color).all():
                out[row][col] = 1
    return out 


def get_observation(world_state, drawer):
    obs = np.zeros((WIDTH, HEIGHT, 3))
    if world_state.is_mission_running:
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')
        if len(world_state.video_frames):
            for frame in reversed(world_state.video_frames):
                if frame.channels == 3:
                    pig_pixels, obs = drawer.showFrame(frame)
                    # scale to between -1, 1
                    return obs
    return obs

def main(agent_host):
    num_episode = 10
    returns = []
    times = []
    drawer = draw_helper()

    for i in range(num_episode):
        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs = get_observation(world_state, drawer)
        episode_return = 0
        next_actions = calc_actions(obs)
        start = time.time()
        # Run episode
        while world_state.is_mission_running:
            # Take step
            for action in next_actions:
                agent_host.sendCommand(action)
            time.sleep(.2)
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("turn 0")
            agent_host.sendCommand("attack 0")
            time.sleep(.1)
            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            obs = get_observation(world_state, drawer)
            next_actions = calc_actions(obs)
            for o in world_state.observations:
                # https://github.com/microsoft/malmo/blob/master/Malmo/samples/Python_examples/hit_test.py
                msg = o.text
                observations = json.loads(msg)
                if all([entity['name'] != 'Pig' for entity in observations['entities']]):
                    agent_host.sendCommand(f"quit")
                    break
                if u'LineOfSight' in observations:
                    los = observations[u'LineOfSight']
                    if los["type"] == "Pig":
                        reward += 30
                        agent_host.sendCommand("attack 1")
                        agent_host.sendCommand("attack 0")
                        time.sleep(0.1)
            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward
        times.append(time.time() - start)
        time.sleep(4)

        num_episode += 1
        returns.append(episode_return)

    print(times)
    # log_returns(returns)


def log_returns(returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(range(len(returns_smooth)), returns_smooth)
    plt.title('Reach the tree')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    s = time.time()
    plt.savefig(f'returns{s}.png')

    with open(f'returns{s}.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 

def calc_actions(obs):
    actions = []
    bin_obs = binary_conv_obs(obs)
    # plt.imshow(bin_obs)
    # plt.show()
    right_side = bin_obs[:, WIDTH//2:]
    left_side = bin_obs[:, :WIDTH//2]
    if np.sum(right_side) > np.sum(left_side):
        actions.append("turn 0.5")
        actions.append("move 1")
    elif np.sum(right_side) < np.sum(left_side):
        actions.append("turn -0.5")
        actions.append("move 1")
    else:
        actions.append("turn -0.5")
    return actions


def init_malmo(agent_host):
        #Record Mission 
        my_mission = MalmoPython.MissionSpec(getXML(obstacles=False), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.setViewpoint(0)
        max_retries = 5
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001))
        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'baseline_seek')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)
        print("done with init malmo")
        return agent_host


if __name__ == '__main__':
    agent_host = MalmoPython.AgentHost()
    main(agent_host)