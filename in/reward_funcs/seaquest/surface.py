def reward_function(self) -> float:
    # +0.1 for moving up (-0.1 for moving down)
    # -10 for losing a life
    lives = None
    player = None
    reward = 0

    for obj in self.objects:
        if 'lives' in str(obj).lower():
            lives = obj
        if 'player' in str(obj).lower():
            player = obj
    
    if player is None or lives is None:
        print("Player or lives object not found")
        return 0
    reward += player.dy / 10 # is y-prev_y
    reward += lives.value_diff * 10 # is (val-prev_val) (-1 if lost a life)

    return reward 
