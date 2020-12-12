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
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
#-----------------------
from LumberjackEnvironment import getXML
from LumberjackQNet import VisionNetwork
from CustomVision import CustomVisionNetwork

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

SIZE = 3 #Dimensions of map
PATH = os.path.join(r'Models', r"state_dict_model%d.pt") #Path to save model
LOAD = False
MODELNUM = 1000

WIDTH = 84
HEIGHT = 84
N_TREES = 10
COLOURS = {'wood': (0, 93, 162), 'leaves':(232, 70, 162), 'grass':(46, 70, 139)}


class Lumberjack(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.drawer = draw_helper()
        self.size = SIZE
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.log_frequency = 10
        self.action_dict = {
            0: 'move',  # Move forward
            1: 'turn', 
        }

        # Rllib Parameters
        self.num_outputs = 2
        #self.action_space = Box(0.0 , 2.00, shape=(2,), dtype=np.float32)
        self.action_space = Box(np.array([-1,-1]), np.array([1,1]),dtype=np.float32)
        self.observation_space = Box(-1.00, 1, shape=(WIDTH, HEIGHT, 3), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()

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
        self.obs, log_pixels  = self.get_observation(world_state)

        return self.obs

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
        reward = 0
        self.agent_host.sendCommand(f"move {(action[0]):30.1f}")
        self.agent_host.sendCommand(f"turn {(action[1]):30.1f}")
        # negative reward for spinning
        reward -= abs(action[1]) * 10000
        reward += abs(action[0]) * 10
        time.sleep(.2)
        self.episode_step += 1

        # Get Done
        world_state = self.agent_host.getWorldState()

        done = False
        if not world_state.is_mission_running:
            done = True
            time.sleep(4)

        # Get Observation

        for error in world_state.errors:
            print("Error:", error.text)
        for o in world_state.observations:
            # https://github.com/microsoft/malmo/blob/master/Malmo/samples/Python_examples/hit_test.py
            msg = o.text
            observations = json.loads(msg)
            if u'LineOfSight' in observations and abs(action[1]) < 0.2:
                los = observations[u'LineOfSight']
                if los["type"] == "Pig":
                    self.agent_host.sendCommand("attack 1")
                    self.agent_host.sendCommand("attack 0")
        self.obs, log_pixels = self.get_observation(world_state) 
        for r in world_state.rewards:
            reward += r.getValue()
        reward += log_pixels/20
        self.episode_return += reward
        # Get Reward


        return self.obs, reward, done, dict()

    def init_malmo(self):
        #Record Mission 
        my_mission = MalmoPython.MissionSpec(getXML(self.max_episode_steps, self.size), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        # my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recording' + str(int(time.time())) + '.tgz']))
        # my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)
        # my_mission.requestVideo(WIDTH, HEIGHT)
        my_mission.setViewpoint(0)

        max_retries = 5
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001)) # add Minecraft machines here as available
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10002))
        # Attempt to start a mission:
        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'wtf')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        obs = np.zeros((WIDTH, HEIGHT, 3))
        if world_state.is_mission_running:
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')
            if len(world_state.video_frames):
                for frame in reversed(world_state.video_frames):
                    if frame.channels == 3:
                        log_pixels = self.drawer.showFrame(frame)
                        pixels = frame.pixels
                        obs = np.reshape(pixels, (WIDTH, HEIGHT, 3)).astype(np.uint8)
                        obs = obs / (255 / 2) - 1
                        # scale to between -1, 1
                        return obs, log_pixels
        return obs, 0
    
    
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
        s = time.time()
        plt.savefig(f'returns{s}.png')

        with open(f'returns{s}.txt', 'w') as f:
            for value in self.returns:
                f.write("{}\n".format(value)) 
# The callback function
def on_postprocess_traj(info):
    """
    arg: {"agent_id": ..., "episode": ...,
        "pre_batch": (before processing),
        "post_batch": (after processing),
        "all_pre_batches": (other agent ids),
    }

    # https://github.com/ray-project/ray/blob/ee8c9ff7320ec6a2d7d097cd5532005c6aeb216e/rllib/policy/sample_batch.py
    Dictionaries in a sample_obj, k:
        t
        eps_id
        agent_index
        obs
        actions
        rewards
        prev_actions
        prev_rewards
        dones
        infos
        new_obs
        action_prob
        action_logp
        vf_preds
        behaviour_logits
        unroll_id       
    """
    agt_id = info["agent_id"]
    eps_id = info["episode"].episode_id
    policy_obj = info["pre_batch"][0]
    sample_obj = info["pre_batch"][1]    
    print('agent_id = {}'.format(agt_id))
    print('episode = {}'.format(eps_id))

    #print("on_postprocess_traj info = {}".format(info))
    #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
    print('actions = {}'.format(sample_obj.columns(["actions"])))
    return


if __name__ == '__main__':
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomVisionNetwork)
    # trainer = ppo.PPOTrainer(env=Lumberjack, config={
    #     'env_config': {},           # No environment parameters to configure
    #     'framework': 'torch',       # Use pyotrch instead of tensorflow
    #     'num_gpus': 0,              # We aren't using GPUs
    #     'num_workers': 2,            # We aren't using parallelism
    #     # Whether to write episode stats and videos to the agent log dir. This is
    #     # typically located in ~/ray_results.
    #     # "monitor": True,
    #     # Set the ray.rllib.* log level for the agent process and its workers.
    #     # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    #     # periodically print out summaries of relevant internal dataflow (this is
    #     # also printed out once at startup at the INFO level). When using the
    #     # `rllib train` command, you can also use the `-v` and `-vv` flags as
    #     # shorthand for INFO and DEBUG.
    #     "log_level": "DEBUG",

    #     # For example, given rollout_fragment_length=100 and train_batch_size=1000:
    #     #   1. RLlib collects 10 fragments of 100 steps each from rollout workers.
    #     #   2. These fragments are concatenated and we perform an epoch of SGD.
    #     "rollout_fragment_length": 400,
    #     # Training batch size, if applicable. Should be >= rollout_fragment_length.
    #     # Samples batches will be concatenated together to a batch of this size,
    #     # which is then passed to SGD.
    #     "train_batch_size": 3200,
    #     "gamma": 0.99,
    #     # Whether to clip rewards during Policy's postprocessing.
    #     # None (default): Clip for Atari only (r=sign(r)).
    #     # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    #     # False: Never clip.
    #     # [float value]: Clip at -value and + value.
    #     # Tuple[value1, value2]: Clip at value1 and value2.
    #     # "clip_rewards": None,
    #     # Whether to clip actions to the action space's low/high range spec.
    #     # "clip_actions": True,
    #     # # Whether to use "rllib" or "deepmind" preprocessors by default
    #     # "explore": True,
    #     # # Provide a dict specifying the Exploration object's config.
    #     # "exploration_config": {
    #     #     # The Exploration class to use. In the simplest case, this is the name
    #     #     # (str) of any class present in the `rllib.utils.exploration` package.
    #     #     # You can also provide the python class directly or the full location
    #     #     # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
    #     #     # EpsilonGreedy").
    #     #     "type": "StochasticSampling",
    #     #     # Add constructor kwargs here (if any).
    #     # },
    #     "preprocessor_pref": "deepmind",
    #     # The default learning rate.
    #     "lr": 0.0001,
    #     # "callbacks": {#"on_episode_start": on_episode_start, 
    #     #                             #"on_episode_step": on_episode_step, 
    #     #                             #"on_episode_end": on_episode_end, 
    #     #                             #"on_sample_end": on_sample_end,
    #     #                             "on_postprocess_traj": on_postprocess_traj,
    #     #                             #"on_train_result": on_train_result,
    #     #                             },
    #     "model": {
    #         "custom_model": "my_model",
    #         "dim": 84, 
    #         "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 1], [64, [5, 5], 1], [32, [42, 42], 1]],
    #         "no_final_linear": True,
    #     }
    # })

    trainer = DDPGTrainer(env=Lumberjack, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 3,            # We aren't using parallelism
        "explore": True,
        "rollout_fragment_length": 128,
        "train_batch_size": 256,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
            "type": "StochasticSampling",
            # "type": "PerWorkerEpsilonGreedy"
            # Add constructor kwargs here (if any).
        },
        "model": {
            "custom_model": "my_model",
            # "dim": 84, 
            "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 1], [64, [5, 5], 1], [32, [42, 42], 1]],
            "no_final_linear": True,
        }
    })

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()

        if i % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
    trainer.save()