from ocatari.ram.seaquest import Player, Diver, Shark, Submarine, PlayerMissile, EnemyMissile, OxygenBar, CollectedDiver

NO_OXYGEN = False
ON_SURFACE = False
LOW_OXYGEN = False
DIVERS = 0
COLLISION = False
COLLECTED = 0


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
    global NO_OXYGEN
    global ON_SURFACE
    global DIVERS
    global COLLISION
    global COLLECTED

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []
    enemies = []
    player_missiles = []
    enemy_missiles = []
    collected_divers = []
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
        elif isinstance(obj, CollectedDiver):
            collected_divers.append(obj)

    if player:
        for enemy in enemies:
            if check_collision(player, enemy): 
                COLLISION = True
        for missile in enemy_missiles:
            if check_collision(player, missile):
                COLLISION = True

    if player and player.y == 46:
        if NO_OXYGEN:
            reward -= 1 # some penalty for drowning
        elif COLLISION:
            reward -= 1 # some penalty for colliding
        elif LOW_OXYGEN: 
            reward += 5 # some reward for surfacing with low oxygen
        elif not ON_SURFACE:
            # surfacing although enough oxygen
            if len(collected_divers) == 6:
                reward += 5
            else: 
                reward -= 1


        # reset flags
        ON_SURFACE = True
        NO_OXYGEN = False
        LOW_OXYGEN = False
        COLLISION = False
    else:
        ON_SURFACE = False

    if oxygen_bar and player and player.y != 46:
        if oxygen_bar.value <= 1:
            NO_OXYGEN = True
        elif oxygen_bar.value < 20: 
            LOW_OXYGEN = True
        else:
            LOW_OXYGEN = False

    return reward