from ocatari.ram.seaquest import Player, Diver

DIVERS = 0
COLLISION = False
COLLECTED = 0
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
        COLLECTED += 1
        COLLISION = False

    if player:
        if player.y > 46:
            ON_SURFACE = False
        elif player.y == 46 and not ON_SURFACE:
            # punish dying and surfacing
            ON_SURFACE = True
            if COLLECTED >= 6:
                reward += 100
            elif COLLECTED > 0:
                reward += 1
            else: 
                reward -= 1

    # update diver count
    DIVERS = len(divers)

    return reward