from ocatari.ram.seaquest import Player, CollectedDiver, OxygenBar

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
    oxygen = None

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, CollectedDiver):
            divers.append(obj)
        elif isinstance(obj, OxygenBar):
            oxygen = obj.value

    if len(divers) > prev_divers:
        reward = 1.0

    if player and oxygen:
        # if player.y > 46:
        if player.y > 52 and oxygen < 64:
            ON_SURFACE = False
            # between 46 and 52 is between the surface and the water
            # if reward != 1.0 and player.y > 52:
            if reward != 1.0:
                reward = 0.001 # small reward for being alive underwater
        elif player.y == 46 and not ON_SURFACE:
            # punish dying and surfacing
            ON_SURFACE = True
            reward = -2.0

    prev_divers = len(divers)

    return reward