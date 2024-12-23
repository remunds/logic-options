prev_player = None

def reward_function(self) -> float:
    # +10 for each enemy killed
    # + 0.01 for moving around unterwater
    # -10 for losing a life

    global prev_player
    player = None
    sharks = []
    subs = []
    missiles = []
    reward = 0

    for obj in self.objects:
        obj_name = str(obj).lower()
        if 'player' in obj_name and 'missile' not in obj_name and 'score' not in obj_name:
            player = obj
        if 'shark' in obj_name:
            sharks.append(obj.xy)
        if 'submarine' in obj_name:
            subs.append(obj.xy)
        if 'playermissile' in obj_name:
            missiles.append(obj.xy)
    
    if player is not None and (player.dy != 0 or player.dx != 0) and player.y > 46: # 46 is surface
        # encourage moving under water
        reward += 0.01

    def hit_shark(missile, shark):
        # 0 is x, 1 is y
        # shark has height 7
        # +/- on x axis depends on direction that we shoot at (and shark travels at 5px/it)
        if missile[0] in range(shark[0]-7, shark[0]+7) and missile[1] in range(shark[1], shark[1]+7):
            return True 
        return False

    def hit_sub(missile, submarine):
        # 0 is x, 1 is y
        # submarine has height 11
        if missile[0] in range(submarine[0]-7, submarine[0]+7) and missile[1] in range(submarine[1], submarine[1]+11):
            return True
        return False


    for missile in missiles:
        for shark in sharks:
            if hit_shark(missile, shark):
                reward += 10 # reward missiles that are close
        for sub in subs:
            if hit_sub(missile, sub):
                reward += 10
        

    if prev_player is not None and player is None: # player is dead
        # player died
        reward -= 100

    prev_player = player
    return reward