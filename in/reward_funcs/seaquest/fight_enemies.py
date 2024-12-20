prev_player = None

def reward_function(self) -> float:
    # +10 for each enemy killed
    # + 0.01 for moving around unterwater
    # -10 for losing a life

    global prev_player
    player = None
    enemies = []
    missiles = []
    reward = 0

    for obj in self.objects:
        obj_name = str(obj).lower()
        if 'player' in obj_name and 'missile' not in obj_name and 'score' not in obj_name:
            player = obj
        if 'shark' in obj_name or 'submarine' in obj_name: 
            enemies.append(obj.xy)
        if 'playermissile' in obj_name:
            missiles.append(obj.xy)
    
    if player is not None and (player.dy != 0 or player.dx != 0) and player.y < 46: # 46 is surface
        # encourage moving under water
        reward += 0.01

    def dist(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    for missile in missiles:
        for enemy in enemies:
            if dist(missile, enemy) < 7:
                reward += 10 # reward missiles that are close

    if prev_player is not None and player is None: # player is dead
        # player died
        reward -= 100

    prev_player = player
    return reward