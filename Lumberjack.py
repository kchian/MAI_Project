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
from numpy.random import randint

SIZE  = 10
OBS_SIZE = 5
MAX_EPISODE_STEPS = 1000
BLOCK = lambda x, y, z, t: "<DrawBlock x='{}'  y='{}' z='{}' type='{}' />".format(x, y, z, t)
CUBOID = lambda x1, x2, y1, y2, z1, z2, t:"<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='{}'/>".format(x1, x2, y1, y2, z1, z2, t)
COLOURS = {'wood': (162, 0, 93), 'leaves':(161, 230, 68), 'grass':(138, 45, 69)}
ACTION_DICT = {
    0: 'move 1',  # Move one block forward
    1: 'turn 1',  # Turn 90 degrees to the right
    2: 'turn -1',  # Turn 90 degrees to the left
    3: 'attack 1'  # Destroy block
}

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
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>MAI Lumberjack</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>8000</StartTime>
                            <AllowPassageOfTime>true</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='0' y2='10' z1='{}' z2='{}' type='air'/>".format(-SIZE-100, SIZE+100, -SIZE-100, SIZE+100) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='-1' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            drawTree(treePos) + \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>MAI Lumberjack</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="0" yaw="0"/>'.format(startX, startZ) + \
                        '''
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_axe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>800</Width>
                            <Height>500</Height>
                        </ColourMapProducer>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <RewardForCollectingItem>
                            <Item type="log" reward="1"/>
                        </RewardForCollectingItem>
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
my_mission_record.setDestination(os.path.sep.join([os.getcwd(), 'recording' + str(int(time.time())) + '.tgz']))
my_mission_record.recordMP4(MalmoPython.FrameType.COLOUR_MAP, 24, 2000000, False)

my_mission.requestVideoWithDepth(800, 500)
my_mission.setViewpoint(0)

# Attempt to start a mission:
max_retries = 1
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)
            continue

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
while world_state.is_mission_running:
    world_state = agent_host.getWorldState()
    # agent_host.sendCommand(random.choice(ACTION_DICT.keys())
    print(".", end="")
    time.sleep(0.1)
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission ended")
# Mission has ended.
