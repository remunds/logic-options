from ocatari.ram.freeway import Chicken

prev_y = 0

def reward_function(self) -> float:
    global prev_y

    game_objects = self.objects
    reward = 0.0

    for obj in game_objects:
        if isinstance(obj, Chicken):
            chicken = obj
            break
    
    if chicken:
        dy = prev_y - chicken.y
        if dy > 0:
            reward = 0.1
        elif dy < 0:
            reward = -0.1
        prev_y = chicken.y

    return reward

