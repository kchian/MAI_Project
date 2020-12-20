from numpy.random import randint

BLOCK = lambda x, y, z, t: "<DrawBlock x='{}'  y='{}' z='{}' type='{}' />".format(x, y, z, t)
CUBOID = lambda x1, x2, y1, y2, z1, z2, t:"<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='{}'/>".format(x1, x2, y1, y2, z1, z2, t)
SIZE = 10
# don't let things spawn on top of the user
default_blocklist = [[i, j] for i in range(-7, 8) for j in range(-7, 8)]
pig_default_blocklist = [[i, j] for i in range(-7, 8) for j in range(-7, 8)]
blocklist = default_blocklist
pig_blocklist = pig_default_blocklist.copy()


def drawTree(coord):
    x, z = coord
    tree = ""
    height = 5
    # tree+=CUBOID(x-1, x+1, height+1, height+1, z-1, z+1, "leaves")
    # tree+=CUBOID(x-2, x+2, height-1, height, z-2, z+2, "leaves")
    for y in range(height):
        #tree+=BLOCK(x, y+2, z, "log")
        tree+=CUBOID(x, x, 2, 4, z, z, "log")
    return tree


def getCoord(bl):
    treePos = [randint(-SIZE + 1, SIZE - 1) for i in range(2)]
    while treePos in bl:
        treePos  = [randint(-SIZE + 1, SIZE - 1) for i in range(2)]
    bl.append(treePos)
    return treePos

def drawPlus(coord):
    block_coords = [(coord[0], coord[1] + 1),
                    (coord[0], coord[1] - 1),
                    (coord[0] + 1, coord[1]),
                    (coord[0] - 1, coord[1])]
    out = ""
    for i in block_coords:
        out += "<DrawBlock x='{}' y='2' z='{}'type='diamond_block'/>".format(*i)
        # blocklist.append(i)

    return out

def drawPig(coord):
    return '<DrawEntity x="{}" y="4" z="{}" type="Pig"/>'.format(coord[0], coord[1])

def drawLava(coord):
    return "<DrawBlock x='{}' y='1' z='{}'type='lava'/>".format(coord[0], coord[1])


def getXML(n_pigs = 5, obstacles = True, missiontype="punch"):
    global blocklist, pig_blocklist
    startX, startZ = (0.5, 0.5)
    inventory = ''
    if missiontype == "kill":
        quit_criteria = '''
            <RewardForMissionEnd rewardForDeath="-5000">
                <Reward description="killed" reward="100" />
                <Reward description="out_of_time" reward="-10000" />
            </RewardForMissionEnd>
            <AgentQuitFromCollectingItem>
                <Item type="porkchop" description="killed"/>
            </AgentQuitFromCollectingItem>
            <AgentQuitFromTimeUp timeLimitMs="40000" description="out_of_time"/>
            '''
        inventory = '''
            <Inventory>
                <InventoryItem slot="0" type="diamond_axe"/>
            </Inventory>
            '''
    else: # if missiontype == "punch"
        quit_criteria = '''
            <RewardForMissionEnd rewardForDeath="-5000">
                <Reward description="out_of_time" reward="00" />
            </RewardForMissionEnd>
            <AgentQuitFromTimeUp timeLimitMs="30000" description="out_of_time"/>
        '''

    out = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <About>
                    <Summary>MAI Lumberjack</Summary>
                </About>
                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>8000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='0' y2='10' z1='{}' z2='{}' type='air'/>".format(-SIZE-100, SIZE+100, -SIZE-100, SIZE+100) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='-3' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='3' z1='{}' z2='{}' type='lapis_block'/>".format(-SIZE - 1, SIZE + 1, -SIZE - 1, SIZE + 1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='3' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "".join(drawTree(getCoord(blocklist)) for coord in range(4) if obstacles) + \
                            "".join(drawPlus(getCoord(blocklist)) for coord in range(4) if obstacles) + \
                            "".join(drawLava(getCoord(blocklist)) for coord in range(4) if obstacles) + \
                            "".join(drawPig(getCoord(pig_blocklist)) for coord in range(n_pigs)) + \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>MAI Lumberjack</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="25" yaw="0"/>'.format(startX, startZ) + \
                        inventory + \
                        '''
                    </AgentStart>
                    <AgentHandlers>
                        <ContinuousMovementCommands>
                            <ModifierList type="allow-list">
                                <command>move</command>
                                <command>turn</command>
                                <command>attack</command>
                            </ModifierList>
                        </ContinuousMovementCommands>
                        <MissionQuitCommands/>
                        <ObservationFromRay/>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>1200</Width>
                            <Height>1200</Height>
                        </ColourMapProducer>
                        <RewardForDamagingEntity>
                            <Mob type="Pig" reward="300"/>
                        </RewardForDamagingEntity>''' + \
                        quit_criteria + \
                        '''
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="300" yrange="60" zrange="60"/>
                        </ObservationFromNearbyEntities>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    blocklist = default_blocklist.copy()
    pig_blocklist = pig_default_blocklist.copy()

    return out
#                             <Block type="log" reward="10.0" behaviour="oncePerTimeSpan" cooldownInMs="0.1"/>
