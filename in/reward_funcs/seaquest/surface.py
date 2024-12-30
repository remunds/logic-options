from ocatari.ram.seaquest import Player, Diver, Shark, Submarine, PlayerMissile, EnemyMissile, OxygenBar

NO_OXYGEN = False
ON_SURFACE = False
LOW_OXYGEN = False
DIVERS = 0
COLLISION = False
COLLECTED = 0


def check_collision(obj1, obj2):
    """
    Check if two GameObjects collide based on their bounding boxes.
    """
    # Calculate boundaries for object A
    right1 = obj1.x + obj1.w + 5
    bottom1 = obj1.y + obj1.h + 5

    # Calculate boundaries for object B
    right2 = obj2.x + obj2.w
    bottom2 = obj2.y + obj2.h

    # Check for overlap on the x-axis
    collision_x = obj1.x < right2 and right1 > obj2.x

    # Check for overlap on the y-axis
    collision_y = obj1.y < bottom2 and bottom1 > obj2.y

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
            reward -= 0.1

        # reset flags
        ON_SURFACE = True
        NO_OXYGEN = False
        LOW_OXYGEN = False
        COLLISION = False
    else:
        ON_SURFACE = False

    if oxygen_bar and player and player.y != 46:
        if oxygen_bar.value == 0:
            NO_OXYGEN = True
        elif oxygen_bar.value < 20: 
            LOW_OXYGEN = True
        else:
            LOW_OXYGEN = False

    return reward