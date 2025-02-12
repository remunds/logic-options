import numpy as np

def meta_policy(oc_state) -> int:
    #oc_state: oc-atari ns-state, shape: (n_envs, buffer_window, state_dim)
    #state_dim = num_objects*2 (x,y)
    option_choices = np.zeros(oc_state.shape[0])
    for i in range(oc_state.shape[0]):
        state = oc_state[i, -1, :]

        # option0: Shooting Enemies
        # option1: Collecting Divers
        # option2: Managing Oxygen

        # Check if collected divers count is 6 or more (indices 37-42)
        collected_divers_idx_start = (1+12+12+4+4+1+1) * 2 + 1 
        collected_divers_idx_end = collected_divers_idx_start + 6*2
        collected_divers = state[collected_divers_idx_start:collected_divers_idx_end]
        divers_count = 0
        for d in collected_divers:
            if d != 0:
                divers_count += 1
        divers_count /= 2
        if divers_count >= 6:
            # print("All 6 divers collected!")
            # Manage oxygen
            option_choices[i] = 2

        # Check oxygen level (index 36)
        ox_bar_idx = (1+12+12+4+4+1+1) * 2 
        oxygen_bar = state[ox_bar_idx]
        if oxygen_bar < 20:
            # print("Oxygen low!")
            # Manage oxygen
            option_choices[i] = 2

        # Idle if player is not present
        player = state[0:2]
        if player.sum() == 0:
            # default to shooting enemies / do nothing
            # print("Player not present!")
            # Shoot Enemies
            option_choices[i] = 0

        px, py = player[0], player[1]
        danger_dist_sq = 50 ** 2  # Squared distance threshold

        # Check Sharks (indices 1-12) and Submarines (indices 13-24)
        shark_idx_start = 1 * 2
        shark_idx_end = shark_idx_start + 12*2
        enemy_sub_idx_start = (1+12) * 2
        enemy_sub_idx_end = enemy_sub_idx_start + 12*2
        x_shark = state[shark_idx_start:shark_idx_end:2]
        y_shark = state[shark_idx_start+1:shark_idx_end:2]
        x_sub = state[enemy_sub_idx_start:enemy_sub_idx_end:2]
        y_sub = state[enemy_sub_idx_start+1:enemy_sub_idx_end:2]
        x_vals = x_shark + x_sub
        y_vals = y_shark + y_sub

        # check both
        for x, y in zip(x_vals, y_vals):
            if x != 0 or y != 0:
                dx = x - px
                dy = y - py
                if (dx ** 2 + dy ** 2) < danger_dist_sq:
                    # Shoot Enemies
                    option_choices[i] = 0

        # Check for any available divers (indices 25-28)
        # this will run whenever no enemy is nearby
        diver_idx_start = (1+12+12) * 2
        diver_idx_end = diver_idx_start + 4*2
        x_diver = state[diver_idx_start:diver_idx_end:2]
        y_diver = state[diver_idx_start+1:diver_idx_end:2]
        for x, y in zip(x_diver, y_diver):
            if x != 0 or y != 0:
                # print("Diver nearby!")
                # Collect Divers
                option_choices[i] = 1
        # for obj in state[25:29]:
        #     if obj.type == 'Diver' and obj.type != 'NoObject':
        #         return 1

        # Default to Shooting Enemies 
        # print("Idling")
        option_choices[i] = 0

    return option_choices