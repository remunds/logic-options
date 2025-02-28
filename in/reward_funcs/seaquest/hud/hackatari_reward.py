from ocatari.ram.seaquest import Player, CollectedDiver, PlayerScore 

ON_SURFACE = True
prev_divers = 0
prev_score = 0

def reward_function(self) -> float:
    global ON_SURFACE
    global prev_divers
    global prev_score

    game_objects = self.objects
    reward = 0.0

    # Define categories for easy identification
    player = None
    divers = []
    score = 0

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, CollectedDiver):
            divers.append(obj)
        elif isinstance(obj, PlayerScore):
            score = obj.value 
    
    if len(divers) > prev_divers:
        reward += 0.5 # reward collecting a diver

    if player:
        if player.y > 46:
            ON_SURFACE = False
            # between 46 and 52 is between the surface and the water
            if reward == 0.0 and player.y > 52:
                reward += 0.001 # small reward for being alive underwater
                if score > prev_score: # reward killing enemies
                    reward += 0.3
        elif player.y == 46 and not ON_SURFACE: # Player surfaces
            # punish dying 
            ON_SURFACE = True
            if len(divers) == 6:
                reward += 1.0
                print("rescued all divers")
            elif len(divers ) > 0:
                reward = 0.0 
            else:
                reward = -2.0 # big punishment for dying/early surfacing

    if score:
        prev_score = score

    prev_divers = len(divers)
    return reward