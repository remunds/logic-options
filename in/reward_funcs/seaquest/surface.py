prev_player_y = None
oxygen_low = False
def reward_function(self) -> float:
    # requires oxygen to be low
    # +0.1 for moving up (-0.1 for moving down)
    # +0.01 for staying down
    # -100 for losing a life

    global prev_player_y, oxygen_low
    player = None
    oxygenbar = None # max is 64
    reward = 0
    divers = 0

    for obj in self.objects:
        obj_name = str(obj).lower()
        if 'oxygenbar' in obj_name and 'depleted' not in obj_name and 'logo' not in obj_name: 
            oxygenbar = obj
        if 'player' in obj_name and 'missile' not in obj_name and 'score' not in obj_name:
            player = obj
        if 'collecteddiver' in obj_name:
            divers += 1

    if oxygenbar is not None:
        # if oxygen is low, encourage moving up
        if oxygenbar.value < 30:
            oxygen_low = True
            if player is not None:
                reward -= player.dy # is y-prev_y
        else:
            oxygen_low = False
            # more than enough oxygen available, encourage staying down
            if player is not None and player.y == 46: # 46 is surface
                reward -= 0.1 # discourage staying on surface

    if prev_player_y is not None and player is None: # player is dead
        # player surfaced with low oxygen -> reward
        if prev_player_y == 46 and oxygen_low:
            reward += 100
        else:
        # either early surfacing, or enemy 
            reward -= 100

    if prev_player_y is not None and player is not None: 
        if prev_player_y > 46 and player.y == 46 and divers > 0:
            # player surfaced with at least one diver (not dying)
            # it's okay, since later the meta policy will decide when to surface
            # here, we just want to learn to surface
            reward += 100

    if player is not None:
        prev_player_y = player.y
    else:
        prev_player_y = None
    return reward 
