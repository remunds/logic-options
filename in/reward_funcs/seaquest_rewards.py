import numpy as np

#TODO: test these reward functions (by playing first)

collected_divers = 0
def reward_function_collect_divers(self) -> float:
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

def reward_function_surface(self) -> float:
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

def reward_function_fight_enemies(self) -> float:
    #TODO: finish implementing
    # +1 for each enemy killed
    # -10 for losing a life

    # we have killed an enemy, if the score increases without being at the surface
    lives = None
    score = None

    for obj in self.objects:
        if 'lives' in str(obj).lower():
            lives = obj
        if 'score' in str(obj).lower():
            score = obj

