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

#imports added for Ray
import gym, ray
from gym.spaces import Box, Discrete
from ray.rllib.agents import ppo
#-----------------------

from LumberjackEnvironment import getXML
from LumberjackQNet import QNetwork
from FrameProcessor import draw_helper

#Hyperparameters
MAX_EPISODE_STEPS = 400
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

SIZE = 50 #Dimensions of map
PATH = os.path.join(r'Models', r"state_dict_model%d.pt") #Path to save model
LOAD = False
MODELNUM = 1000

WIDTH = 800
HEIGHT = 500
N_TREES = 10
COLOURS = {'wood': (0, 93, 162), 'leaves':(232, 70, 162), 'grass':(46, 70, 139)}

class Lumberjack(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = SIZE
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.log_frequency = 10
        self.action_dict = {
            0: 'move',  # Move forward
            1: 'turn', 
        }

        # Rllib Parameters
        self.action_space = Box(-1, 1, shape=(len(self.action_dict), ))
        self.observation_space = Box(0, 255, shape=(np.prod([4, 800, 500]), ), dtype=np.uint8)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:

            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Lumberjack Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        for i, j in self.action_dict.items():
            self.agent_host.sendCommand(f"{j} {action[i]:30.1f}")
        time.sleep(.2)
        self.episode_step += 1

        # Get Done
        world_state = self.agent_host.getWorldState()

        done = False
        if self.episode_step >= self.max_episode_steps or not world_state.is_mission_running:
            done = True

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def init_malmo(self):
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
                self.agent_host.startMission( my_mission, my_mission_record )
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)
                    continue
        return self.agent_host

    def get_observation(self, world_state):
        obs = np.zeros((4, 800, 500))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
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

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(10) / 10
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('Reach the tree')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for value in self.returns:
                f.write("{}\n".format(value)) 

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=Lumberjack, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0,            # We aren't using parallelism

        # Whether to write episode stats and videos to the agent log dir. This is
        # typically located in ~/ray_results.
        "monitor": True,
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level). When using the
        # `rllib train` command, you can also use the `-v` and `-vv` flags as
        # shorthand for INFO and DEBUG.
        "log_level": "WARN",

        # Training batch size, if applicable. Should be >= rollout_fragment_length.
        # Samples batches will be concatenated together to a batch of this size,
        # which is then passed to SGD.
        "train_batch_size": 200,
        "gamma": 0.99,
        # Whether to clip rewards during Policy's postprocessing.
        # None (default): Clip for Atari only (r=sign(r)).
        # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
        # False: Never clip.
        # [float value]: Clip at -value and + value.
        # Tuple[value1, value2]: Clip at value1 and value2.
        "clip_rewards": None,
        # Whether to clip actions to the action space's low/high range spec.
        "clip_actions": True,
        # Whether to use "rllib" or "deepmind" preprocessors by default
        "preprocessor_pref": "rllib",
        # The default learning rate.
        "lr": 0.0001,
    })

    while True:
        print(trainer.train())
