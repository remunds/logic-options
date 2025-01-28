from ocatari.ram.seaquest import Player, Diver, Shark, Submarine, PlayerMissile, EnemyMissile, OxygenBar

ON_SURFACE = False
NO_OXYGEN = False
COLLISION = False
NON_SURFACE_COUNTER = 0

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
    global ON_SURFACE
    global NO_OXYGEN
    global COLLISION
    global NON_SURFACE_COUNTER
    MAX_COUNTER = 2081
    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    enemies = []
    enemy_missiles = []
    oxygen_bar = None

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Shark) or isinstance(obj, Submarine):
            enemies.append(obj)
        elif isinstance(obj, EnemyMissile):
            enemy_missiles.append(obj)
        elif isinstance(obj, OxygenBar):
            oxygen_bar = obj

    if player:
        for enemy in enemies:
            if check_collision(player, enemy): 
                COLLISION = True
        for missile in enemy_missiles:
            if check_collision(player, missile):
                COLLISION = True

    if player and player.y == 46:
        # player on surface, either by surfacing, colliding or drowning
        # this event triggers the reward calculation
        reward = NON_SURFACE_COUNTER / MAX_COUNTER
        if NO_OXYGEN or COLLISION:
            # reward /= 3  # not surfacing on time / colliding
            reward = -1 + reward   # the longer the player stays underwater, the less negative the reward
        # if not ON_SURFACE and reward < 0.1:
        #     reward = -0.10

        # reset flags
        NON_SURFACE_COUNTER = 0
        ON_SURFACE = True
        NO_OXYGEN = False
        COLLISION = False
    else:
        ON_SURFACE = False

    if player and player.y != 46:
        NON_SURFACE_COUNTER += 1

    if oxygen_bar and player and player.y != 46:
        if oxygen_bar.value <= 1:
            NO_OXYGEN = True

    return reward