from ocatari.ram.seaquest import Player, Shark, Submarine, PlayerMissile

ENEMIES = 0
COLLISION = False
ON_SURFACE = True

def check_collision(obj1, obj2):
    """
    Check if two GameObjects collide based on their bounding boxes.
    """
    # Calculate boundaries for object A
    right1 = obj1.x + obj1.w + 10
    bottom1 = obj1.y + obj1.h + 10

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