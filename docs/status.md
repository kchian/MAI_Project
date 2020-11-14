---
layout: default
title: Status
---
## Watch the Status Report
[![Watch the Status Report](https://img.youtube.com/vi/kHjt8DJ0yOw/maxresdefault.jpg)](https://youtu.be/kHjt8DJ0yOw) 

## Project Summary
Steve will start in a 10x10 grass block platform. There will be at least one tree on this platform for Steve to mine. After all, we've heard that he would like to become a lumberjack. There are several iterative goals for this project. Our first goal is to have Steve find a tree. He may be facing in a direction where there are no trees so Steve will have to turn and find a tree. Our next goal is for Steve to walk towards the tree. You can't mine the tree before you reach it! Finally, we want Steve to mine the tree. Steve will get his input as rgb pixels using ColourMapProducer. We hope to prepare Steve to become a lumberjack worthy of competing with Paul Bunyon.

## Approach
For our general approach we have used a Q Network to determine the best action given the environment. The network is updated using adaptive moment estimation (ADAM). Adam uses estimates of the first and second moments of the gradient. This is done using mt=β1mt−1+(1−β1)gt for the first moment and vt=β2vt−1+(1−β2)g^2t for the second moment.

The network takes the rgb input as 3 channels of 800 by 500. The input is then modified through a series of two convolultional layers with pooling, dropout, and two fully conneted layers. The network gives 3 different values as output, each correlating with a different action. The 3 actions are "move 0.5", "turn 0.25", and "turn -0.25". We figured that the agent wouldn't need to stop when moving towards the tree and wanted to simplify the model as much as possible so we chose to stick with just these three outputs. The agent is allowed to move continiously through the environment using these actions. The networks recieves negative rewards when the agent moves off the platform and positive rewards for when the agent looks at or touches the tree. The network also receives a reward based on the distance between the agent and the tree.

We have tried several modifications to improve our performance. One modification was to add more trees to the environment and slowly reduce the number of trees as training continues. The rational behind this decision was that using a single tree means that the agent isn't very likely to come in contact with the tree. By increasing the reward, the agent will come in contact with more trees which could reduce the amount of time needed for the agent to make the correlation between a tree and a positive reward. Another modification was to remove pooling and simply use larger kernel sizes in the convolutional layers. This modification reflects the current trend in state of the art systems. We also tried adding a barrier with a negative reward to the edge of the environment so that the agent would be able to train longer instead of falling off the side. One of the more effective modifications we made was to the rewards provided to the network. At first we had a -10000 reward for falling off the edge and a 10000 reward for touching the tree. This led to the agent prioritizing staying on the platform instead of trying to reach the tree. Instead of having a moving agent we instead had an agent that would spin in circles for the duration of each episode. After lowering the reward for falling off the edge to -200, we were able to get a better performing agent. 

## Evaluation
# An important aspect of your project, as we mentioned in the beginning, is evaluating yourproject. Be clear and precise about describing the evaluation setup, for both quantitative and qualitativeresults. Present the results to convince the reader that you have aworkingimplementation. Use plots, charts,tables, screenshots, figures, etc. as needed. I expect you will need at least a 1-2 paragraphs to describe eachtype of evaluation that you perform.


## Remaining Goals and Challenges
# In a few paragraphs, describe your goals for the next 4-5 weeks, whenthe final report is due. At the very least, describe how you consider your prototype to be limited, and whatyou want to add to make it a complete contribution. Note that if you think your algorithm is quite good,but have not performed sufficient evaluation, doing them can also be a reasonable goal. Similarly, you maypropose some baselines (such as a hand-coded policy) that you did not get a chance to implement, butwant to compare against for the final submission. Finally, given your experience so far, describe some of thechallenges you anticipate facing by the time your final report is due, how crippling you think it might be,and what you might do to solve them

## Resources used
# Mention all the resources that you found useful in writing your implementation. This shouldinclude everything like code documentation, AI/ML libraries, source code that you used, StackOverflow, etc.You do not have to be comprehensive, but it is important to report the ones that are crucial to your project. Iwould like to know these so that the more useful ones can be shared with others in the course
