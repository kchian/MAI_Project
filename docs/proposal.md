---
layout: default
title:  Proposal
---

# Proposal

## Summary of the Project
The agent will start on a small floating platform  (10x10 grass block platform) with a single tree.The agent will need to find the tree by using ColourMapProducer, walk towards the tree, and then touch the tree. After that, we will move the agent into a larger platform (50x50) filled with a forest of trees and surrounded by a lapis block wall. Here the amount of trees will diminish over time. The agent needs to still be able to find and touch a tree as they get sparcer. Finally, we will place the agent in a pen (25x25) with 1 pig. Here the agent will need to chase and kill the pig.

## AI/ML Algorithms
Reinforcement learning with neural function approximator

## Evaluation Plan
In the first stage, we will evaluate our AI purely on its ability to stay on the platform and touch the tree. The baseline would be an AI that doesn’t do anything meaning that it can’t find the tree, or walk towards the tree. Our AI will first have to find the tree. The AI may have to turn around to find the tree. Once it finds the tree it should be rewarded. Once the AI finds the tree a couple of times it should be a fairly simple task.
After it passes the first stage, we will move onto the next. In the forest stage, the AI will be rewarded for getting closer to the tree and will be penalized for getting further away from the tree. It will also be penalized for touching the lapis wall. We expect that after a dozen or so runs in this new environment, the AI will be able to adjust and begin walking towards trees with great accuracy.
Finally, we will need to tweak the AI a bit so it can prioritize pigs and not trees. However, after we made the change, we should be able to drop the AI into a pig pen so it can hunt pigs. At first, we expect the AI to randomly wander around and swing it’s sword. Over time it should be able to keep the pig in the center of the screen and walk towards it. Once close enough, it should kill the pig.
We will evaluate the AI primarily through visual inspection on the first stage. In the second stage, we will calculate an average reward score over time. We expect the score to rise in the beginning before settling for a bit and then dipping as the amount of trees gets sparser before settling again. In the final stage, we will time how long it takes for the AI to kill the pig. There will also be a time limit that ends the mission if the AI takes too long. It will probably hit the time limit in the beginning but slowly takes less time to kill the pig
