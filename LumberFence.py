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

def createMarker(index, coord):
    return '<Marker name="Tree' + str(index) + '" x="'+str(coord[0])+'" y="0" z="'+str(coord[1])+'"/>'

def getTree(blocklist, SIZE):
    treePos = [randint(-SIZE, SIZE) for i in range(2)]
    while treePos in blocklist:
        treePos  = [randint(-SIZE, SIZE) for i in range(2)]
    return treePos

def getXML(MAX_EPISODE_STEPS, SIZE, N_TREES):
    
    startX, startZ = [randint(-SIZE, SIZE) for i in range(2)]
    blocklist = [[startX, startZ]]
    trees = []
    for i in range(N_TREES):
        treePos = getTree(blocklist, SIZE)
        blocklist.append(treePos)
        trees.append(treePos)
    startX+=0.5
    startZ+=0.5
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
                            "<DrawCuboid x1='{}' x2='{}' y1='-3' y2='-1' z1='{}' z2='{}' type='grass'/>".format(-SIZE*2, SIZE*2, -SIZE*2, SIZE*2) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='-3' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='3' z1='{}' z2='{}' type='lapis_block'/>".format(-SIZE-1, SIZE+1, -SIZE-1, SIZE+1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='3' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "".join(drawTree(coord) for coord in trees) + \
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
                        <RewardForTouchingBlockType>
                            <Block type="log" reward="100"/>
                            <Block type="lapis_block" reward="-1"/>
                        </RewardForTouchingBlockType>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <RewardForMissionEnd>
                            <Reward description="found tree" reward="0"/>
                        </RewardForMissionEnd>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="300" yrange="60" zrange="60"/>
                        </ObservationFromNearbyEntities>
                        <ObservationFromDistance>''' + \
                            ''.join(createMarker(index, coord) for index, coord in enumerate(trees)) + \
                            '''
                        </ObservationFromDistance>
                        <AgentQuitFromTouchingBlockType>
                            <Block type="log"/>
                        </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    
