collected_divers = 0
def reward_function(self) -> float:
    # +1 reward for collecting a diver
    # -10 for losing a life
    global collected_divers
    current_divers = 0
    lives = None
    reward = 0

    for obj in self.objects:
        if 'collecteddiver' in str(obj).lower():
            current_divers += 1
        if 'lives' in str(obj).lower():
            lives = obj

    reward += current_divers
    reward -= collected_divers

    if lives is not None:
        reward += lives.value_diff * 10 # is (val-prev_val) (-1 if lost a life)

    collected_divers = current_divers

    return reward
