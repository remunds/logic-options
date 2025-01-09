from ocatari.ram.seaquest import Player, Shark, Submarine, PlayerMissile

ENEMIES = 0
COLLISION = False
ON_SURFACE = True

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
    global ENEMIES
    global COLLISION
    global ON_SURFACE

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    enemies = []
    player_missiles = []

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Shark) or isinstance(obj, Submarine):
            enemies.append(obj)
        elif isinstance(obj, PlayerMissile):
            player_missiles.append(obj)

    if player:
        for missile in player_missiles:
            for enemy in enemies:
                if check_collision(missile, enemy):
                    COLLISION = True

    if ENEMIES > len(enemies) and COLLISION:
        reward += 1
        COLLISION = False

    if player:
        if player.y > 46:
            ON_SURFACE = False
        elif player.y == 46 and not ON_SURFACE: # Player surfaces
            # punish dying and surfacing
            ON_SURFACE = True
            reward -= 1

        
    ENEMIES = len(enemies)

    return reward