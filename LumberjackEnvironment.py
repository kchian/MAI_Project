from numpy.random import randint

BLOCK = lambda x, y, z, t: "<DrawBlock x='{}'  y='{}' z='{}' type='{}' />".format(x, y, z, t)
CUBOID = lambda x1, x2, y1, y2, z1, z2, t:"<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='{}'/>".format(x1, x2, y1, y2, z1, z2, t)

def drawTree(coord):
    x, z = coord
    tree = ""
    height = 5
    # tree+=CUBOID(x-1, x+1, height+1, height+1, z-1, z+1, "leaves")
    # tree+=CUBOID(x-2, x+2, height-1, height, z-2, z+2, "leaves")
    for y in range(height):
        #tree+=BLOCK(x, y+2, z, "log")
        tree+=CUBOID(x-1, x+1, 2, 5, z-1, z+1, "log")
    return tree

def createMarker(index, coord):
    return '<Marker name="Tree' + str(index) + '" x="'+str(coord[0])+'" y="0" z="'+str(coord[1])+'"/>'

def getTree(blocklist, SIZE):
    treePos = [randint(-SIZE, SIZE) for i in range(2)]
    while treePos in blocklist:
        treePos  = [randint(-SIZE, SIZE) for i in range(2)]
    return treePos

def getXML(MAX_EPISODE_STEPS, SIZE = 5, N_TREES = 10):
    startX, startZ = [randint(-SIZE, SIZE) for i in range(2)]
    startX = 15
    
    blocklist = [[i, j] for i in range(startX - 1, startX + 2) for j in range(startZ - 1, startZ + 2) ]
    trees = []
    for i in range(N_TREES):
        treePos = getTree(blocklist, SIZE)
        blocklist.append(treePos)
        trees.append(treePos)
    startX+=0.5
    startZ+=0.5
    
    SIZEX = 2
    SIZEZ = 9
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                            "<DrawCuboid x1='{}' x2='{}' y1='-3' y2='1' z1='{}' z2='{}' type='grass'/>".format(-100, 100, -100, 100) + \
                            "".join([
                                "<DrawEntity x='{}' y='2' z='{}' type='Pig'/>".format(randint(-100, 100) + 0.5, randint(-100, 100) + 0.5)
                                for i in range(250)
                            ]) + \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>MAI Lumberjack</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="30" yaw="45"/>'.format(startX, startZ) + \
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
                                <command>attack</command>
                            </ModifierList>
                        </ContinuousMovementCommands>
                        <ObservationFromRay/>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>84</Width>
                            <Height>84</Height>
                        </ColourMapProducer>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="8" yrange="2" zrange="6" />
                        </ObservationFromNearbyEntities>
                        <RewardForDamagingEntity>
                            <Mob type="Pig" reward="1000"/>
                        </RewardForDamagingEntity>
                        <RewardForMissionEnd rewardForDeath="-1">
                            <Reward description="out_of_time" reward="00" />
                        </RewardForMissionEnd>
                        <AgentQuitFromTimeUp timeLimitMs="30000" description="out_of_time"/>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    
#                             <Block type="log" reward="10.0" behaviour="oncePerTimeSpan" cooldownInMs="0.1"/>
