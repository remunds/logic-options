def reward_function(self) -> float:
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
