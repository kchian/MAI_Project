---
layout: default
title:  Proposal
---

# Proposal

## Summary of the Project
Steve will start in an enclosed area (10x10 grass block platform) with a single tree somewhere on the platform. Under the platform is raised high up in the air so if Steve steps off the platform, he dies. Steve will need to find the tree by using ColourMapProducer and DepthProducer. Then, he will need to walk towards the tree. And Finally, he will need to mine up the trunk of the tree.

## AI/ML Algorithms
Reinforcement learning with neural function approximator

## Evaluation Plan
We will evaluate our AI through its ability to find a tree and mine it. The baseline would be an AI that doesn’t do anything meaning that it can’t find the tree, walk towards the tree, or attempt to mine the tree. The AI will first have to find the tree. The AI may have to turn around to find the tree. Once it finds the tree it should be rewarded. Once the AI finds the tree a couple of times it should be a fairly simple task. From there the AI will be rewarded for getting closer to the tree and will be penalized for getting further away from the tree. We expect the AI to be able to walk towards a tree with great accuracy. After the AI reaches the tree, it will be rewarded for mining the tree. We will reward the AI for hitting the tree in order to incentivize the AI to mine the tree. We expect this to be a little more difficult than finding the tree but believe we can achieve it through training.
	We can monitor our AI by watching it perform. We should first see the AI find the tree in the environment. We should then be able to see the AI walk towards the tree. Finally we should be able to see the AI mine the tree. We will know the AI is not performing well when it fails to achieve any one of these tasks. Our moonshot case is an AI that can quickly and efficiently find and mine a tree.

