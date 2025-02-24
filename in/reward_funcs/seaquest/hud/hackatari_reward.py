from ocatari.ram.seaquest import Player, CollectedDiver, Shark, Submarine, PlayerMissile, EnemyMissile, OxygenBar

COLLISION_ENEMY = False
ON_SURFACE = True
HIT_ENEMY = False
prev_enemies = 0
prev_divers = 0

def check_collision(player, obj, margin=3):
    """
    Check if two GameObjects collide based on their bounding boxes.
    """
    # (x, y) is the top left corner of the object
    # Calculate boundaries for object A
    left1 = player.x - margin 
    right1 = player.x + player.w + margin
    top1 = player.y - margin
    bottom1 = player.y + player.h + margin

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
    global COLLISION_ENEMY
    global ON_SURFACE
    global HIT_ENEMY
    global prev_enemies
    global prev_divers

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []
    enemies = []
    player_missiles = []
    enemy_missiles = []

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, CollectedDiver):
            divers.append(obj)
        elif isinstance(obj, Shark) or isinstance(obj, Submarine):
            enemies.append(obj)
        elif isinstance(obj, PlayerMissile):
            player_missiles.append(obj)
        elif isinstance(obj, EnemyMissile):
            enemy_missiles.append(obj)
    
    if len(divers) > prev_divers:
        reward += 0.3 # reward collecting a diver

    if player:
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

    if prev_enemies > len(enemies) and HIT_ENEMY:
        reward += 0.1  # reward hitting an enemy
        HIT_ENEMY = False

    if player:
        if player.y > 46:
            ON_SURFACE = False
            # between 46 and 52 is between the surface and the water
            if reward == 0.0 and player.y > 52:
                reward += 0.001 # small reward for being alive underwater
        elif player.y == 46 and not ON_SURFACE:
            if len(divers) == 6:
                reward += 10 # reward for surfacing with all divers
                print("All 6 divers collected and surfaced!")
            elif len(divers) > 0:
                reward = 0.0
            else:
                reward -= 1 # punish dying 
            ON_SURFACE = True

    prev_divers = len(divers)
    prev_enemies = len(enemies)

    return reward