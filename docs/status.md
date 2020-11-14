---
layout: default
title: Status
---
## Watch the Status Report
[![Watch the Status Report](https://img.youtube.com/vi/kHjt8DJ0yOw/maxresdefault.jpg)](https://youtu.be/kHjt8DJ0yOw) 

## Project Summary
Steve will start in a 10x10 grass block platform. There will be at least one tree on this platform for Steve to mine. After all, we've heard that he would like to become a lumberjack. There are several iterative goals for this project. Our first goal is to have Steve find a tree. He may be facing in a direction where there are no trees so Steve will have to turn and find a tree. Our next goal is for Steve to walk towards the tree. You can't mine the tree before you reach it! Finally, we want Steve to mine the tree. Steve will get his input as rgb pixels using ColourMapProducer. We hope to prepare Steve to become a lumberjack worthy of competing with Paul Bunyon.

## Approach
For our general approach we have used a Q Network to determine the best action given the environment. The network is updated using adaptive moment estimation (ADAM). Adam uses estimates of the first and second moments of the gradient. 

![Moment](https://github.com/kchian/ForkThePork/images/moment.png)

β is the rate of decay for the past estimates. The moments are then divided by 1 - β in order to correct for bias. 

![Correction](https://github.com/kchian/ForkThePork/images/biasCorrection.png)

Finally, the values are used to update each weight of the network.

![Update](https://github.com/kchian/ForkThePork/images/update.png)

The network takes the rgb input as 3 channels of 800 by 500. The input is then modified through a series of two convolultional layers, dropout, and two fully conneted layers. The network gives 3 different values as output, each correlating with a different action. The 3 actions are "move 0.5", "turn 0.25", and "turn -0.25". We figured that the agent wouldn't need to stop when moving towards the tree and wanted to simplify the model as much as possible so we chose to stick with just these three outputs. The agent is allowed to move continiously through the environment using these actions. The networks recieves negative rewards when the agent moves off the platform and positive rewards for when the agent looks at or touches the tree. The network also receives a reward based on the distance between the agent and the tree.

We have tried several modifications to improve our performance. One modification was to add more trees to the environment and slowly reduce the number of trees as training continues. The rational behind this decision was that using a single tree means that the agent isn't very likely to come in contact with the tree. By increasing the reward, the agent will come in contact with more trees which could reduce the amount of time needed for the agent to make the correlation between a tree and a positive reward. Another modification was to remove pooling and simply use larger kernel sizes in the convolutional layers. This modification reflects the current trend in state of the art systems. We also tried adding a barrier with a negative reward to the edge of the environment so that the agent would be able to train longer instead of falling off the side. One of the more effective modifications we made was to the rewards provided to the network. At first we had a -10000 reward for falling off the edge and a 10000 reward for touching the tree. This led to the agent prioritizing staying on the platform instead of trying to reach the tree. Instead of having a moving agent we instead had an agent that would spin in circles for the duration of each episode. After lowering the reward for falling off the edge to -200, we were able to get a better performing agent. 

## Evaluation

The most important result that we evaluated our agent on was success rate. The success rate is the percentage of trials that were completed by the agent touching the tree. In a single tree environment a random agent has less than a 2% success rate. Our best agent was able to reach a 7% success rate. While there is a lot more progress to be made, we are encouraged by the imporvement in our agent. In the multitree environment with 10 trees a random agent reaches has a little more than a 3% success rate. We were able to train an agent with around a 70% success rate.

![Tree](https://github.com/kchian/ForkThePork/images/tree.png) 

Another way we evaluated our agent was through its death rate. This evaluation method was only applicable to the agent in the environment with no wall. The death rate is the percentage of trials where the agent falls off the edge of the map. A random agent falls off the map almost 90% of the time. Our trained model only falls off the map a little more than 10% of the time. This evaluation method is a little flawed in determining how the agent is progressing towards the goal of reaching the tree. In some instances, we have observed the agent spinning in circles instead of moving around, presumably in order to avoid falling to its death.

![Death](https://github.com/kchian/ForkThePork/images/death.png)

The last quantitative way we evaluated our agent was through its return values. A higher return value means the agent is doing well and a lower return value means it is not doing as well. The average return value was not such a great indicator of performance since the positive reward for touching the tree was very high and would greatly skew the reward. Instead, we looked through the history of rewards and looked to see if we could find a higher concentration of positive rewards further into training.

We evaluated our agent qualitatively by watching it in the environment. We watched to see if our agent seemed to find the tree or just look around aimelessly. We watched to see if the agent would move towards the tree instead of away from it. We watched to see what happened when our agent got closer to the tree. We wathced to see if our agent spend a lot of time falling off the edges or walking into walls.

![Wall](https://github.com/kchian/ForkThePork/images/wall.png)

The qualitative analysis gave us a lot of insight into our agents performance. When training our agent on a platform we noticed that it would spin in circles to avoid falling off the edge. When the agent was placed closer to the tree it would pan back and forth to get the reward for looking at the tree but it would not take the initiative to move on its own and relied on a random decision to move in order to do so. These observations prompted us to tweak the rewards which led to better performance in our agent. 

*Many of our rate are rounded. This was done due to the lack of evaluation trials run for the agents. Most agents were evaluated in less than 200 episodes.

## Remaining Goals and Challenges
# In a few paragraphs, describe your goals for the next 4-5 weeks, whenthe final report is due. At the very least, describe how you consider your prototype to be limited, and whatyou want to add to make it a complete contribution. Note that if you think your algorithm is quite good,but have not performed sufficient evaluation, doing them can also be a reasonable goal. Similarly, you maypropose some baselines (such as a hand-coded policy) that you did not get a chance to implement, butwant to compare against for the final submission. Finally, given your experience so far, describe some of thechallenges you anticipate facing by the time your final report is due, how crippling you think it might be,and what you might do to solve them



## Resources used
-Malmo
-Pytorch
-Malmo Pig Example 
https://github.com/microsoft/malmo-challenge/tree/master/ai_challenge/pig_chase
-CNN and Pooling
https://d2l.ai/chapter_convolutional-neural-networks/pooling.html
