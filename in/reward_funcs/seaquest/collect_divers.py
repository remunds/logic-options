from ocatari.ram.seaquest import Player, Diver

DIVERS = 0
COLLISION = False
COLLECTED = 0
ON_SURFACE = True


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
    global DIVERS
    global COLLISION
    global COLLECTED
    global ON_SURFACE

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Diver):
            divers.append(obj)

    if player:
        for diver in divers:
            # detect diver collection
            if check_collision(player, diver): 
                COLLISION = True

    if DIVERS > len(divers) and COLLISION:
        # if previous diver count is higher than current diver count
        # and a collection was detected -> actually collected a diver
        reward += 1  # Scaled down reward for collecting a diver
        COLLECTED += 1
        COLLISION = False

    if player:
        if player.y > 46:
            ON_SURFACE = False
        elif player.y == 46 and not ON_SURFACE:
            # punish dying and surfacing
            ON_SURFACE = True
            reward -= 1

    # update diver count
    DIVERS = len(divers)

    return reward