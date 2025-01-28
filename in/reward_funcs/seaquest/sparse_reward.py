from ocatari.ram.seaquest import Player, CollectedDiver

COLLISION_ENEMY = False
COLLECTED = 0
PREV_COLLECTED = 0
ON_SURFACE = True
PLAYER = False

# Requires HUD (CollectedDiver) to be enabled

def reward_function(self) -> float:
    global COLLECTED
    global PREV_COLLECTED
    global ON_SURFACE
    global PLAYER

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    collected_divers = []

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, CollectedDiver):
            collected_divers.append(obj)

    if player:
        PLAYER = True

    if not player: 
        if PLAYER:
            reward -= 10
        PLAYER = False


    # problem: if player collects two divers, surfaces (+2 reward) and then submerges and surface again, reward is +1 again.
    if player:
        if player.y > 46:
        # player is underwater
            ON_SURFACE = False
            COLLECTED = len(collected_divers)
        elif player.y == 46 and not ON_SURFACE:
        # player just surfaced 
            if COLLECTED == 6:
                reward += 100
            elif COLLECTED > 0: 
                reward += COLLECTED - PREV_COLLECTED
                PREV_COLLECTED = COLLECTED - 1 
            ON_SURFACE = True
            COLLECTED = 0

    return reward