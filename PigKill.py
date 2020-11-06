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
from numpy.random import randint

SIZE  = 10
OBS_SIZE = 5
MAX_EPISODE_STEPS = 100
BLOCK = lambda x, y, z, t: "<DrawBlock x='{}'  y='{}' z='{}' type='{}' />".format(x, y, z, t)
CUBOID = lambda x1, x2, y1, y2, z1, z2, t:"<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='{}'/>".format(x1, x2, y1, y2, z1, z2, t)

def drawTree(coord):
    x, z = coord
    tree = ""
    height = 5
    tree+=CUBOID(x-1, x+1, height+1, height+1, z-1, z+1, "leaves")
    tree+=CUBOID(x-2, x+2, height-1, height, z-2, z+2, "leaves")
    for y in range(height):
        tree+=BLOCK(x, y+2, z, "log")
    return tree

def getXML():
    treePos = [randint(-SIZE, SIZE) for i in range(2)]
    startX, startZ   = [randint(-SIZE, SIZE) for i in range(2)]
    while treePos==[startX, startZ]:
        startX, startZ   = [randint(-SIZE, SIZE) for i in range(2)]

#Mission step-up came from https://github.com/microsoft/malmo-challenge/blob/master/ai_challenge/pig_chase/pig_chase.xml
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>Catch the pig!</Summary>
                </About>

                <ModSettings>
                    <MsPerTick>4</MsPerTick>
                </ModSettings>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>6000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                        <AllowSpawning>false</AllowSpawning>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;minecraft:bedrock,2*minecraft:dirt,minecraft:grass;1;village"/>
                        <DrawingDecorator>
                            <!-- Tricks to clean the map before drawing (avoid entity duplication on the map) -->
                            <!-- coordinates for cuboid are inclusive -->
                            <DrawCuboid x1="-10" y1="4" z1="-10" x2="10" y2="45" z2="10" type="air"/>

                            <!-- Area Limits -->
                            <DrawLine x1="1" y1="3" z1="0" x2="7" y2="3" z2="0" type="sand"/>
                            <DrawLine x1="1" y1="4" z1="0" x2="7" y2="4" z2="0" type="fence"/>

                            <DrawLine x1="1" y1="3" z1="6" x2="7" y2="3" z2="6" type="sand"/>
                            <DrawLine x1="1" y1="4" z1="6" x2="7" y2="4" z2="6" type="fence"/>

                            <DrawLine x1="1" y1="3" z1="0" x2="1" y2="3" z2="2" type="sand"/>
                            <DrawLine x1="1" y1="4" z1="0" x2="1" y2="4" z2="2" type="fence"/>
                            <DrawLine x1="0" y1="3" z1="2" x2="0" y2="3" z2="4" type="sand"/>
                            <DrawLine x1="0" y1="4" z1="2" x2="0" y2="4" z2="4" type="fence"/>
                            <DrawLine x1="1" y1="3" z1="4" x2="1" y2="3" z2="6" type="sand"/>
                            <DrawLine x1="1" y1="4" z1="4" x2="1" y2="4" z2="6" type="fence"/>

                            <DrawLine x1="7" y1="3" z1="0" x2="7" y2="3" z2="2" type="sand"/>
                            <DrawLine x1="7" y1="4" z1="0" x2="7" y2="4" z2="2" type="fence"/>
                            <DrawLine x1="8" y1="3" z1="2" x2="8" y2="3" z2="4" type="sand"/>
                            <DrawLine x1="8" y1="4" z1="2" x2="8" y2="4" z2="4" type="fence"/>
                            <DrawLine x1="7" y1="3" z1="4" x2="7" y2="3" z2="6" type="sand"/>
                            <DrawLine x1="7" y1="4" z1="4" x2="7" y2="4" z2="6" type="fence"/>

                            <!-- Path blocker -->
                            <DrawBlock x="3" y="3" z="2" type="sand"/>
                            <DrawBlock x="3" y="4" z="2" type="fence"/>

                            <DrawBlock x="3" y="3" z="4" type="sand"/>
                            <DrawBlock x="3" y="4" z="4" type="fence"/>

                            <DrawBlock x="5" y="3" z="2" type="sand"/>
                            <DrawBlock x="5" y="4" z="2" type="fence"/>

                            <DrawBlock x="5" y="3" z="4" type="sand"/>
                            <DrawBlock x="5" y="4" z="4" type="fence"/>

                            <DrawBlock x="1" y="3" z="3" type="lapis_block"/>
                            <DrawBlock x="7" y="3" z="3" type="lapis_block"/>

                            <!-- Pig -->
                            <DrawEntity x="4.5" y="4" z="3.5" type="Pig"/>

                        </DrawingDecorator>
                        <ServerQuitFromTimeUp timeLimitMs="1000000"/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>MAI Lumberjack</Name>
                    <AgentStart>
                        <Placement x="2.5" y="4" z="5.5" pitch="30" yaw="180"/>
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_axe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>320</Width>
                            <Height>240</Height>
                        </ColourMapProducer>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <RewardForCollectingItem>
                            <Item type="log" reward="1"/>
                        </RewardForCollectingItem>
                        <RewardForTouchingBlockType>
                            <Block type="lava" reward="-1"/>
                        </RewardForTouchingBlockType>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    
######################################################################################
######################################################################################

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(getXML(), True)
my_mission_record = MalmoPython.MissionRecordSpec()
my_mission.requestVideo(800, 500)
my_mission.setViewpoint(1)

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

steps = 0
# Loop until mission ends:
while world_state.is_mission_running and steps < MAX_EPISODE_STEPS:
    steps+=1
    print(".", end="")
    time.sleep(0.1)
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission ended")
# Mission has ended.