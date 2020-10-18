---
layout: default
title:  Proposal
---

# {{ page.title }}

##Summary of the Project
Our AI will spawn in a cave biome and its main goal is to mine as much valuable ore as possible. It will be given a pickaxe and points are awarded for each ore mined. Rarer ores, such as diamond and redstone, are awarded more points. It will be able to explore the cave and avoid hazards such as lava, water, and pitfalls. Its surroundings will be provided through Malmo's framework, and target ores will be provided as input parameters. If it finds ore, it will stop, mine, and pick up the ore. The output will be an inventory full of ores.

##AI/ML Algorithms
Reinforcement learning with neural function approximator

##Evaluation Plan
We will evaluate our AI through the materials it is able to mine. The AI will get more points for mining more valuable ores and will be penalized slightly for mining less valuable materials such as stone and dirt. The AI will also be penalized when it takes damage or dies. The AI will be allowed to mine until the pickaxe fails. The baseline will be negative when the AI doesn’t do anything. We expect that the total value of the AI’s inventory will drastically increase through training. 
We will be able to monitor whether our AI works by watching it mine. We expect to see the AI find valuable ores, move towards them, and mine them. We also expect to see the AI spend a lot of time traveling and exploring the cave. Our moonshot case would be an unstoppable AI that leaves no ores behind.
