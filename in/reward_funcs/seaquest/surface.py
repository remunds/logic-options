prev_player = None
oxygen_low = False
def reward_function(self) -> float:
    # requires oxygen to be low
    # +0.1 for moving up (-0.1 for moving down)
    # -100 for losing a life

    global prev_player, oxygen_low
    player = None
    oxygenbar = None # max is 64
    reward = 0

    for obj in self.objects:
        if 'oxygenbar' in str(obj).lower() and 'depleted' not in str(obj).lower():
            oxygenbar = obj
        if 'player' in str(obj).lower() and 'missile' not in str(obj).lower() and 'score' not in str(obj).lower():
            player = obj
    
    if oxygenbar is not None:
        # if oxygen is low, encourage moving up
        if oxygenbar.value < 30:
            oxygen_low = True
            if player is not None:
                reward -= player.dy / 10 # is y-prev_y
        else:
            oxygen_low = False
            # more than enough oxygen available, encourage staying down
            if player.y > 46: # 46 is surface
                reward += 0.01

    if prev_player is not None and player is None: # player is dead
        # player surfaced with low oxygen -> reward
        if prev_player.y == 46 and oxygen_low:
            reward += 100
        else:
        # either early surfacing, or enemy 
            reward -= 100


    prev_player = player 
    return reward 
