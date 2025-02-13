from ocatari.ram.seaquest import Player, CollectedDiver, OxygenBar

ON_SURFACE = True
PREV_PLAYER_POS = 0

def reward_function(self) -> float:
    global ON_SURFACE
    global PREV_PLAYER_POS

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []
    oxygen = 0

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, CollectedDiver):
            divers.append(obj)
        elif isinstance(obj, OxygenBar):
            oxygen = obj.value

    if player:
        if player.y > 46:
            ON_SURFACE = False
            if oxygen <= 10 or len(divers) >= 6:
                reward -= 5
        elif player.y == 46 and not ON_SURFACE:
            if (oxygen <= 10 or len(divers) >= 6) and PREV_PLAYER_POS <= 48: # ensure did not drown or collide
                reward += 100
            else:
                reward -= 1 
            ON_SURFACE = True

        PREV_PLAYER_POS = player.y

    return reward