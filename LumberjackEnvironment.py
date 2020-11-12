from numpy.random import randint

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

def getXML(MAX_EPISODE_STEPS = 1000, SIZE  = 10):
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
                        <ContinuousMovementCommands>
                            <ModifierList type="allow-list">
                                <command>move</command>
                                <command>turn</command>
                            </ModifierList>
                        </ContinuousMovementCommands>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>800</Width>
                            <Height>500</Height>
                        </ColourMapProducer>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <RewardForMissionEnd rewardForDeath="-100000">
                            <Reward description="found tree" reward="10000"/>
                        </RewardForMissionEnd>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="300" yrange="60" zrange="60"/>
                        </ObservationFromNearbyEntities>
                        <ObservationFromDistance>
                            <Marker name="Tree" x="'''+str(treePos[0])+'''" y="0" z="'''+str(treePos[1])+'''"/>
                        </ObservationFromDistance>
                        <AgentQuitFromTouchingBlockType>
                            <Block type="log"/>
                        </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    