from ocatari.ram.seaquest import Player, Diver, Shark, Submarine, PlayerMissile, EnemyMissile, OxygenBar

LOW_OXYGEN = False
DIVERS = 0
COLLISION_DIVER = False
COLLISION_ENEMY = False
COLLECTED = 0
ON_SURFACE = True
HIT_ENEMY = False
ENEMIES = 0

def check_collision(player, obj):
    """
    Check if two GameObjects collide based on their bounding boxes.
    """
    # (x, y) is the top left corner of the object
    # Calculate boundaries for object A
    left1 = player.x - 2
    right1 = player.x + player.w + 2
    top1 = player.y - 2
    bottom1 = player.y + player.h + 2

    # Calculate boundaries for object B
    left2 = obj.x
    right2 = obj.x + obj.w
    top2 = obj.y
    bottom2 = obj.y + obj.h

    # Check for overlap on the x-axis
    collision_x = left1 < right2 and right1 > left2

    # Check for overlap on the y-axis
    collision_y =  top1 < bottom2 and bottom1 > top2

    # Return True if both conditions are met, otherwise False
    return collision_x and collision_y


def reward_function(self) -> float:
    global LOW_OXYGEN
    global DIVERS
    global COLLISION_DIVER  
    global COLLISION_ENEMY
    global COLLECTED
    global ON_SURFACE
    global HIT_ENEMY
    global ENEMIES

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []
    enemies = []
    player_missiles = []
    enemy_missiles = []
    oxygen_bar = None

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Diver):
            divers.append(obj)
        elif isinstance(obj, Shark) or isinstance(obj, Submarine):
            enemies.append(obj)
        elif isinstance(obj, PlayerMissile):
            player_missiles.append(obj)
        elif isinstance(obj, EnemyMissile):
            enemy_missiles.append(obj)
        elif isinstance(obj, OxygenBar):
            oxygen_bar = obj

    if player:
        for diver in divers:
            if check_collision(player, diver) and COLLECTED != 6:
                COLLISION_DIVER = True
        for enemy in enemies:
            if check_collision(player, enemy):
                COLLISION_ENEMY = True
        for missile in enemy_missiles:
            if check_collision(player, missile):
                COLLISION_ENEMY = True
        for missile in player_missiles:
            for enemy in enemies:
                if check_collision(missile, enemy):
                    HIT_ENEMY = True

    if DIVERS > len(divers) and COLLISION_DIVER:
        reward += 1  # reward for collecting a diver
        COLLECTED += 1
        COLLISION_DIVER = False
    DIVERS = len(divers)

    if ENEMIES > len(enemies) and HIT_ENEMY:
        reward += 1  # reward hitting an enemy
        HIT_ENEMY = False
    ENEMIES = len(enemies)

    if player:
        if player.y > 46:
            ON_SURFACE = False
        elif player.y == 46 and not ON_SURFACE:
            if COLLISION_ENEMY:
                reward -= 1
                COLLISION_ENEMY = False
            elif COLLECTED == 6:
                reward += 100
                COLLECTED = 0
            elif COLLECTED > 0:
                COLLECTED -= 1
                if LOW_OXYGEN:
                    reward += 10
                else:
                    reward -= 0.1 # punish early surfacing
            else:
                reward -= 1 # punish dying and early surfacing
            ON_SURFACE = True
            LOW_OXYGEN = False

    # if player and player.y == 46:
    #     if COLLECTED == 6:
    #         reward += 100 # high reward for surfacing with all divers
    #         COLLECTED = 0
    #     elif COLLECTED > 0:
    #         COLLECTED -= 1
    #         if LOW_OXYGEN:
    #             reward += 5 # some reward for surfacing with low oxygen
    #         else:
    #             reward += 0 
    #     LOW_OXYGEN = False

    if oxygen_bar and player and player.y != 46:
        if oxygen_bar.value < 20: 
            LOW_OXYGEN = True
        else:
            LOW_OXYGEN = False

    # currently rewards: collection of divers and low-oxygen surfacing
    # want also: avoid dying, reward enemy kills


    return reward