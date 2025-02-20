import numpy as np

def meta_policy(oc_state) -> int:
    #oc_state: oc-atari ns-state, shape: (n_envs, buffer_window, state_dim)
    #state_dim = num_objects*2 (x,y)

    option_choices = np.zeros(oc_state.shape[0])
    for i in range(oc_state.shape[0]):
        state = oc_state[i, -1, :]

        chicken_y = state[1] 
        # option0: lower part
        # option1: upper part 
        if chicken_y > 100:
            option_choices[i] = 1
        else:
            option_choices[i] = 0
    return option_choices