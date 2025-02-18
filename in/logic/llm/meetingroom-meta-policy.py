import torch

def meta_policy(state_batch: torch.Tensor) -> int:
    # state:
    #[star(bfxy) - player(bfxy) # 4
    # elevator(xy) - player(xy) #2
    # entrance(xy) - player(xy) #2
    # player(f) # 1
    # playerview # view_size*view_size = 7*7=49
    # total: 58

    # three options:
    # 0: navigate to elevator
    # 1: navigate to star within correct level
    # 2: use elevator to go to correct level 


    # Extract components from batched state tensor
    goal_floor_diff = state_batch[:, 1]    # Floor difference from goal
    elevator_x_diff = state_batch[:, 4]    # Elevator X distances
    elevator_y_diff = state_batch[:, 5]    # Elevator Y distances
    
    # Initialize decisions to "navigate to star" (1) as default
    decisions = torch.ones(state_batch.size(0), 
                          dtype=torch.int32,
                          device=state_batch.device)
    
    # Create boolean masks for efficient decision logic
    not_on_target_floor = (goal_floor_diff != 0)
    at_elevator = (elevator_x_diff == 0) & (elevator_y_diff == 0)
    
    # Set elevator usage (2) where needed
    elevator_mask = not_on_target_floor & at_elevator
    decisions[elevator_mask] = 2
    
    # Set elevator navigation (0) where needed
    nav_to_elevator_mask = not_on_target_floor & ~at_elevator
    decisions[nav_to_elevator_mask] = 0

    return decisions

    # Prompt:
    # Implement a RL meta-policy for the meeting room environment (specified further below)
    # by writing a function called meta_policy that retrieves the current state of the environment.
    # The state consists of the following:
    # - Distance of player to the goal (4-dimensional vector: [building, floor, x, y])
    # - Distance of player to the elevator (current level, so 2-dimensional vector: [x, y])
    # - Distance of player to the entrance (current level, so 2-dimensional vector: [x, y])
    # - The floor the player is on (1-dimensional), floor
    # - The player's view (7x7 grid, 49-dimensional)
    # All as a single 58-dimensional vector.
    # The state uses batching, so its actually (B, 58) where B is the batch size.

    # The game is a grid-based game where the player navigates through a building with multiple floors.
    # The floors are connected by an elevator. The player's goal is to reach a star on the correct floor.
    # The player can move in four directions north, south, east, and west and can use the elevator to change floors (next, prev).
    # The player's view is a 7x7 grid centered around the player's position, but can only see the next three tiles. 

    # The meta-policy has access to sub-policies (options) and should return one of three options (as int):
    # 0: Navigate to the elevator
    # 1: Navigate to the star within the correct level
    # 2: Use the elevator to go to the correct level
    # 
    # Find reasonable conditions to choose between these options based on the state information provided. 
    # Return the meta-policy function as a python snippet.