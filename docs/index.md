---
layout: default
title:  Home
---

[Source code](https://github.com/kchian/MAI_Project)

Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)

Our team spawns in a discord server every Monday at 10AM.

The goal is to train a reinforcement learning agent to locate, track, and reach a pig around obstacles like trees and lava.
  
![The Target](images/pig.png) ![The Destruction](images/pig2.png)
  
As a first step, we evaluated our AI purely on its ability to stay on the platform and touch a tree we place in a 10x10 floating world. The baseline in this scenario would be an AI that doesn’t do anything and wanders aimlessly. Our AI uses Malmo's ColourMap Video Producer to get RGB frames where each block type is assigned a unique color. These frames were used to determine whether the agent was looking towards a tree, and an appropriate reward was given. The AI would ideally learn to turn around to find the tree, similar to a radar. Once it finds the tree, walks towards it, and touches it, the mission ends in success. 

As a variation of this, the AI was placed in a 50x50 world with 10 trees, enclosed by a lapis-lazuli block wall. The agent was rewarded for getting closer to the tree and penalized for getting further away from the tree, alongside the tree-locating reward from the first step. It was also penalized for touching the lapis-lazuli wall. We expect that some time in this new environment, the AI will be able to adjust and begin walking towards trees with great accuracy. The increased number of trees allows for a more consistent locating ability by the agent.

Since the framework is already up and running, next steps to get the agent up and hunting pigs include increasing action space with more perspective turning, introduction of some controlled obstacles, and an improved convolutional neural network function approximator to help the agent understand its observations.
