prev_player = None
collected_divers = 0
def reward_function(self) -> float:
    # +10 reward for collecting a diver
    # +0.1 for moving (under water)
    # -10 for losing a life

    global prev_player, collected_divers
    player = None
    current_divers = 0
    reward = 0

    for obj in self.objects:
        obj_name = str(obj).lower()
        if 'collecteddiver' in obj_name: 
            current_divers += 1
        if 'player' in obj_name and 'missile' not in obj_name and 'score' not in obj_name:
            player = obj

    if current_divers > collected_divers: 
        reward += 10

    if player is not None and (player.dy != 0 or player.dx != 0) and player.y > 46: # 46 is surface
        # encourage moving under water
        reward += 0.01


    if prev_player is not None and player is None: # player is dead
        reward = -100
    
    # print(player.visible)
    # if abs(player.dy) > 10:
    #     # dying alternative
    #     reward -= 100

    collected_divers = current_divers
    prev_player = player

    return reward
