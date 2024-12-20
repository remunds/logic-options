prev_player = None
collected_divers = 0
def reward_function(self) -> float:
    # +1 reward for collecting a diver
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

    reward += current_divers
    reward -= collected_divers

    if prev_player is not None and player is None: # player is dead
        reward -= 10

    collected_divers = current_divers
    prev_player = player

    return reward
