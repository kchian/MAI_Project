from numpy.random import randint

BLOCK = lambda x, y, z, t: "<DrawBlock x='{}'  y='{}' z='{}' type='{}' />".format(x, y, z, t)
CUBOID = lambda x1, x2, y1, y2, z1, z2, t:"<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='{}'/>".format(x1, x2, y1, y2, z1, z2, t)

def getXML(MAX_EPISODE_STEPS = 1000, SIZE  = 50, n = 0):
    startX, startZ = [randint(-SIZE, SIZE) for i in range(2)]
    blockStartX, blockStartZ = [randint(-SIZE, SIZE) for i in range(2)]
    while blockStartX==startX and startZ==blockStartZ:
        blockStartX, blockStartZ = [randint(-SIZE, SIZE) for i in range(2)]
    startX+=0.5
    startZ+=0.5
    blockStartX+=0.5
    blockStartZ+=0.5
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
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='5' z1='{}' z2='{}' type='lapis_block'/>".format(-SIZE-1, SIZE+1, -SIZE-1, SIZE+1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='5' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            '<DrawEntity x="{}" y="2" z="{}" type="Pig"/>'.format(blockStartX, blockStartZ) + \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>MAI Lumberjack</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="30" yaw="0"/>'.format(startX, startZ) + \
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
                        <ObservationFromFullStats/>
                        <ObservationFromRay/>
                        <ColourMapProducer>
                            <Width>800</Width>
                            <Height>500</Height>
                        </ColourMapProducer>
                        <RewardForDamagingEntity>
                            <Mob type="Pig" reward="1000"/>
                        </RewardForDamagingEntity>
                        <RewardForCollectingItem>
                             <Item type="porkchop" reward="10000"/>
                        </RewardForCollectingItem>
                        <AgentQuitFromCollectingItem>
                            <Item type="porkchop"/>
                        </AgentQuitFromCollectingItem>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                        <RewardForMissionEnd>
                            <Reward description="found tree" reward="0"/>
                        </RewardForMissionEnd>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


