from ocatari.ram.seaquest import Player, CollectedDiver, OxygenBar

ON_SURFACE = True
PREV_PLAYER_POS = 0

def reward_function(self) -> float:
    # simply reward moving up, surfacing and punish dying
    # only makes sense, if start is restricted by condition on state 
    global ON_SURFACE
    global PREV_PLAYER_POS

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj

    if player:
        if player.y > 46:
            ON_SURFACE = False
            if PREV_PLAYER_POS > player.y:
                # moved up
                reward = 0.01 #small reward for moving up
        elif player.y == 46 and not ON_SURFACE:
            # surfaced by itself or died
            if PREV_PLAYER_POS <= 48:
                reward = 1.0 # big reward for surfacing
            else:
                reward = -1.0 # big punishment for drowning/colliding 
            ON_SURFACE = True
            print("dead surface")

        PREV_PLAYER_POS = player.y

    return reward