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
from PigCatchSmall import getXML
from LumberjackQNet import VisionNetwork
from CustomVision import CustomVisionNetwork
from FCNet import FCNet

from FrameProcessor import draw_helper

LOAD = True
TRAIN = False
WIDTH, HEIGHT = (20, 20)
pig_color = np.array([1, 57, 110])
N_PIGS = 5


def binary_conv_obs(obs):
    out = np.zeros((WIDTH, HEIGHT))
    for row in range(WIDTH):
        for col in range(HEIGHT):
            if (obs[row][col] == pig_color).all():
                out[row][col] = 1
    return out 

class Lumberjack(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.drawer = draw_helper()
        self.log_frequency = 10
        self.action_dict = {
            0: 'move',  # Move forward
            1: 'turn', 
        }

        # Rllib Parameters
        self.num_outputs = 2
        #self.action_space = Box(0.0 , 2.00, shape=(2,), dtype=np.float32)
        self.action_space = Box(np.array([0, -0.5]), np.array([0.75, 0.5]),dtype=np.float32)
        # self.observation_space = Box(-1.00, 1, shape=(WIDTH,HEIGHT,3), dtype=np.float32)
        self.observation_space = Box(-1.00, 1, shape=(WIDTH*HEIGHT*3,), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()

        # Lumberjack Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.times = []
        self.killed_pigs = []
        self.seen = {}
        self.stats = {
            'success': [],
            'timeout': [],
            'death': [],
        }


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
        self.start = time.time()
        self.seen = {}
        
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
        self.agent_host.sendCommand("attack 0")
        
        # negative reward for spinning
        # reward -= abs(action[0]) * 10
        # reward -= abs(action[1]) * 10
        reward -= 20
        # Try upping this
        time.sleep(0.2)
        self.agent_host.sendCommand(f"move 0")
        self.agent_host.sendCommand(f"turn 0")
        self.episode_step += 1
        # Get Done
        world_state = self.agent_host.getWorldState()
        
        done = False
        if not world_state.is_mission_running:
            duration = time.time() - self.start
            if duration >= 30:
                self.stats['timeout'].append(duration)
            else:
                self.stats['death'].append(duration)
            self.times.append(duration)
            for o in world_state.observations:
                msg = o.text
                observations = json.loads(msg)
                if 'entities' in observations:
                    self.killed_pigs.append(sum([1 for entity in observations['entities'] if entity['name'] == 'Pig']))
            reward += 600 - duration * 15
            done = True
            time.sleep(4)

        # Get Observation

        for error in world_state.errors:
            print("Error:", error.text)
        for o in world_state.observations:
            # https://github.com/microsoft/malmo/blob/master/Malmo/samples/Python_examples/hit_test.py
            msg = o.text
            observations = json.loads(msg)
            # if no more pigs left
            if 'entities' in observations:
                if all([entity['name'] != 'Pig' for entity in observations['entities']]):
                    done = True
                    duration = time.time() - self.start
                    self.stats['success'].append(duration)
                    self.agent_host.sendCommand(f"quit")
                    time.sleep(4)
                    self.times.append(duration)
                    self.killed_pigs.append(N_PIGS)
                    reward += 600 - duration * 15
                    break
                for e in observations['entities']:
                    if e['name'] == 'agent':
                        coords = (np.floor(e['x']), np.floor(e['z']))
                        if coords not in self.seen:
                            self.seen[coords] = 0
                            reward += 50
                        self.seen[coords] += 3
                        reward -= self.seen[coords]
                        reward += e['z'] / 2
            if u'LineOfSight' in observations:
                los = observations[u'LineOfSight']
                if los["type"] == "Pig":
                    self.agent_host.sendCommand("attack 1")
                    self.agent_host.sendCommand("attack 0")
        self.obs, pixels = self.get_observation(world_state) 
        for r in world_state.rewards:
            reward += r.getValue()
        reward += (1 - np.exp(-pixels)) * 30

        self.episode_return += reward
        # Get Reward
        return self.obs, reward, done, dict()

    def init_malmo(self):
        print("doing init malmo")
        #Record Mission 
        my_mission = MalmoPython.MissionSpec(getXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        # my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recording' + str(int(time.time())) + '.tgz']))
        # my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)
        # my_mission.requestVideo(WIDTH, HEIGHT)
        my_mission.setViewpoint(0)
        print("adding to client pool")
        max_retries = 5
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
        # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001)) # add Minecraft machines here as available
        # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10002))
        # Attempt to start a mission:
        print("attempting to start a mission")
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
        print("done with init malmo")
        return world_state

    def get_observation(self, world_state):
        obs = np.zeros((WIDTH, HEIGHT, 3))
        if world_state.is_mission_running:
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')
            if len(world_state.video_frames):
                for frame in reversed(world_state.video_frames):
                    if frame.channels == 3:
                        pig_pixels, obs = self.drawer.showFrame(frame)
                        # pixels = frame.pixels
                        obs = obs / (255 / 2) - 1
                        # scale to between -1, 1
                        return obs.flatten(), pig_pixels
        return obs.flatten(), 0
    
    
    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(10) / 10
        returns_smooth = np.convolve(self.returns, box, mode='same')
        times_smooth = np.convolve(self.times, box, mode='same')

        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('1 layer')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        s = time.time()
        plt.savefig(f'returns{s}.png')
        plt.clf()

        with open(f'returns{s}.txt', 'w') as f:
            for value in self.returns:
                f.write("{}\n".format(value)) 
        plt.plot(range(len(times_smooth)), times_smooth)
        plt.savefig(f'times{s}.png')


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
    ModelCatalog.register_custom_model("my_model", FCNet)
    
    trainer = ppo.PPOTrainer(env=Lumberjack, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 2,            # We aren't using parallelism
        # Whether to write episode stats and videos to the agent log dir. This is
        # typically located in ~/ray_results.
        # "monitor": True,
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level). When using the
        # `rllib train` command, you can also use the `-v` and `-vv` flags as
        # shorthand for INFO and DEBUG.
        "log_level": "DEBUG",
        "vf_clip_param": 500,
        # For example, given rollout_fragment_length=100 and train_batch_size=1000:
        #   1. RLlib collects 10 fragments of 100 steps each from rollout workers.
        #   2. These fragments are concatenated and we perform an epoch of SGD.
        "rollout_fragment_length": 64,
        "sgd_minibatch_size": 64,

        # Training batch size, if applicable. Should be >= rollout_fragment_length.
        # Samples batches will be concatenated together to a batch of this size,
        # which is then passed to SGD.
        "train_batch_size": 128,
        "gamma": 0.999,
        # Whether to clip rewards during Policy's postprocessing.
        # None (default): Clip for Atari only (r=sign(r)).
        # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
        # False: Never clip.
        # [float value]: Clip at -value and + value.
        # Tuple[value1, value2]: Clip at value1 and value2.
        # "clip_rewards": None,
        # Whether to clip actions to the action space's low/high range spec.
        # "clip_actions": True,
        # # Whether to use "rllib" or "deepmind" preprocessors by default
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "StochasticSampling",
            # Add constructor kwargs here (if any).
        },
        "preprocessor_pref": "deepmind",
        # The default learning rate.
        "lr": 0.001,
        # "callbacks": {#"on_episode_start": on_episode_start, 
        #                             #"on_episode_step": on_episode_step, 
        #                             #"on_episode_end": on_episode_end, 
        #                             #"on_sample_end": on_sample_end,
        #                             "on_postprocess_traj": on_postprocess_traj,
        #                             #"on_train_result": on_train_result,
        #                             },
        "model": {
            "custom_model": "my_model",
            # "dim": 84, 
            # "fcnet_hiddens": [800]
            # # Used to be 42, 42 to get it to the right shape
            # "conv_filters": [[16, [4, 4], 2], 
            #                  [32, [16, 16], 2], 
            #                  [64, [5, 5], 2], 
            #                  [128, [16, 16], 1], 
            #                  [128, [16, 16], 1],
            #                  [128, [11, 11], 1]],
            # "no_final_linear": True,
        }
    })

    if LOAD:
        # this is the checkpoint from something trained in a small environment with a single pig
        # trainer.restore(r"C:\Users\Kevin\Documents\classes\CS175\checkpoints\turn_withpunch_linear\checkpoint_171\check")
        trainer.restore(r"C:\Users\Kevin\Documents\classes\CS175\checkpoints\small_set_obstacles\best\checkpoint_412\check")
    if TRAIN:
        for i in range(1000):
            # Perform one iteration of training the policy with PPO
            result = trainer.train()

            if i % 10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
        trainer.save()
    else:
        # instantiate env class
        env = Lumberjack({})
        for i in range(50):
            # run until episode ends
            episode_reward = 0
            done = False
            obs = env.reset()
            while not done:
                action = trainer.compute_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
        print(f"stats:{env.stats}")
        print(f"times:{env.times}")