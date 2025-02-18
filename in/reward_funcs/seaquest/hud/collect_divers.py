from ocatari.ram.seaquest import Player, CollectedDiver

ON_SURFACE = True
prev_divers = 0

def reward_function(self) -> float:
    global ON_SURFACE
    global prev_divers

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, CollectedDiver):
            divers.append(obj)

    if len(divers) > prev_divers:
        reward = 1.0

    if player:
        if player.y > 46:
            ON_SURFACE = False
            if reward != 1.0:
                reward = 0.001 # small reward for being alive underwater
        elif player.y == 46 and not ON_SURFACE:
            # punish dying and surfacing
            ON_SURFACE = True
            reward = -1.0

    prev_divers = len(divers)

    return reward