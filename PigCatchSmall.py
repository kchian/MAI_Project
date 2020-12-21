from numpy.random import randint

BLOCK = lambda x, y, z, t: "<DrawBlock x='{}'  y='{}' z='{}' type='{}' />".format(x, y, z, t)
CUBOID = lambda x1, x2, y1, y2, z1, z2, t:"<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='{}'/>".format(x1, x2, y1, y2, z1, z2, t)
def getTree(blocklist, SIZE):
    treePos = [randint(-SIZE, SIZE) for i in range(2)]
    while treePos in blocklist:
        treePos  = [randint(-SIZE, SIZE) for i in range(2)]
    return treePos

SIZEX = 2
SIZEZ = 9

def drawTree(x, z):
    return CUBOID(x, x, 2, 2, z, z, "log")


def drawLava(x, z):
    return "<DrawBlock x='{}' y='1' z='{}' type='lava'/>".format(x, z)


def drawObstacles():
    out = ""
    # for z in range(2, 7, 2):
    #     x = randint(-SIZEX, SIZEX)
    #     obstacle = randint(0, 2)
    #     for i in range(2):
    #         x = randint(-SIZEX, SIZEX)
    #         if obstacle == 0:
    #             out += drawLava(x, z)
    #         else:
    #             out += drawTree(x, z)
    
    lava = [(-2, 5), (0, 5), (1, 5), (3, 5)]
    for x, z in lava:
        out += drawLava(x, z)
        
    tree = [(-2, 3), (0, 3), (2, 3)]
    for x, z in tree:
        print(x, z)
        out += drawTree(x, z)
    # for z in range(2, 7, 2):
    #     x = randint(-SIZEX, SIZEX)
    #     obstacle = randint(0, 2)
    #     for i in range(2):
    #         x = randint(-SIZEX, SIZEX)
    #         if obstacle == 0:
    #             out += drawLava(x, z)
    #         else:
    #             out += drawTree(x, z)
    return out


def getXML():
    startX, startZ = (0, 0)
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
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='0' y2='10' z1='{}' z2='{}' type='air'/>".format(-100, 100, -100, 100) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='-3' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZEX - 1, SIZEX + 1, -1, SIZEZ + 1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='lapis_block'/>".format(-SIZEX - 1, SIZEX + 1, -1, SIZEZ + 1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='air'/>".format(-SIZEX, SIZEX, 0, SIZEZ) + \
                            drawObstacles() + \
                            '''
                            <DrawCuboid x1='-1' x2='1' y1='2' y2='2' z1='8' z2='9' type='fence'/>
                            <DrawCuboid x1='0' x2='0' y1='2' y2='2' z1='9' z2='9' type='air'/>
                            <DrawCuboid x1='-1' x2='-1' y1='2' y2='2' z1='8' z2='8' type='air'/>
                            <DrawCuboid x1='1' x2='1' y1='2' y2='2' z1='8' z2='8' type='air'/>
                            <DrawEntity x="0.5" y="2" z="9.5" type="Pig"/>
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>agent</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="30" yaw="0"/>'.format(startX, startZ) + \
                        '''
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_axe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <MissionQuitCommands/>
                        <ObservationFromRay/>
                        <ObservationFromFullStats/>
                        <ColourMapProducer>
                            <Width>1200</Width>
                            <Height>1200</Height>
                        </ColourMapProducer>
                        <RewardForDamagingEntity>
                            <Mob type="Pig" reward="300"/>
                        </RewardForDamagingEntity>
                        <RewardForMissionEnd rewardForDeath="-300">
                            <Reward description="out_of_time" reward="-1000" />
                        </RewardForMissionEnd>
                        <AgentQuitFromTimeUp timeLimitMs="30000" description="out_of_time"/>
                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="300" yrange="60" zrange="60"/>
                        </ObservationFromNearbyEntities>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''
    