# self_play.py

import chess
import numpy as np
import os
import datetime
from tqdm import tqdm
# Import MCTS and your model components
from mcts import mcts_search, get_game_result_value # Removed encode_policy_target from here
from NNKOVA3 import board_to_tensor, build_alpha_zero_style_model # Assuming you'll import relevant parts
from mcts_utils import encode_policy_target, TOTAL_MOVE_CHANNELS # This is the correct source for encode_policy_target

def generate_self_play_data(model, num_games: int, output_dir: str, num_simulations: int, policy_output_size=4672):
    os.makedirs(output_dir, exist_ok=True)
    
    game_data = [] # List to store (board_tensor, policy_target_array, game_result_value)

    for i in tqdm(range(num_games), desc="Generating Self-Play Games"):
        board = chess.Board()
        current_game_states = []
        current_game_policy_targets = []

        while not board.is_game_over():
            # Run MCTS to get the best move and policy probabilities
            # temp=1.0 for initial exploration in early games
            # temp=0.0 for later games to exploit learned policy
            policy_target_map = mcts_search(board.copy(), model, num_simulations=num_simulations, temperature=1.0) # Adjust temperature

            # Store current board state and MCTS-derived policy target
            current_game_states.append(board_to_tensor(board.copy()))
            current_game_policy_targets.append(encode_policy_target(policy_target_map, board.copy(), policy_output_size))

            # Select move based on policy_target_map (e.g., sample proportionally to probs)
            # For simplicity now, let's just pick the max (you might want to sample)
            if not policy_target_map: # No moves to pick, shouldn't happen if game not over
                break
            best_move_uci = max(policy_target_map, key=policy_target_map.get)
            chosen_move = chess.Move.from_uci(best_move_uci)
            
            if chosen_move not in board.legal_moves:
                # This indicates an issue with policy mapping or MCTS
                print(f"Illegal move selected by MCTS: {chosen_move.uci()} for board: {board.fen()}")
                # Fallback: pick a random legal move to continue
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    chosen_move = np.random.choice(legal_moves)
                else:
                    break # No legal moves, game should be over

            board.push(chosen_move)

        # Game is over, determine result and propagate to all states in the game
        game_result_value = get_game_result_value(board) # 1.0 (white win), -1.0 (black win), 0.0 (draw)

        # AlphaZero flips the board for Black's perspective
        for idx, board_tensor in enumerate(current_game_states):
            policy_target_array = current_game_policy_targets[idx]
            
            # If it was black's turn in that state, flip board and value
            # You might need to adjust board_to_tensor to handle colors properly for consistent NN input
            # And policy_target_array would need to be flipped too.
            # This is complex, so for initial setup, we might skip explicit board flipping for training data
            # and just rely on the network learning both perspectives.
            
            game_data.append((board_tensor, policy_target_array, game_result_value))

    # Save collected data
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    inputs_path = os.path.join(output_dir, f"self_play_inputs_{timestamp}.npy")
    policy_targets_path = os.path.join(output_dir, f"self_play_policy_targets_{timestamp}.npy")
    value_targets_path = os.path.join(output_dir, f"self_play_value_targets_{timestamp}.npy")

    # Separate inputs, policy targets, and value targets
    X_data = np.array([item[0] for item in game_data])
    Policy_data = np.array([item[1] for item in game_data])
    Value_data = np.array([item[2] for item in game_data]).reshape(-1, 1)

    np.save(inputs_path, X_data)
    np.save(policy_targets_path, Policy_data)
    np.save(value_targets_path, Value_data)

    print(f"Generated {len(game_data)} self-play samples.")
    print(f"Saved inputs to {inputs_path}")
    print(f"Saved policy targets to {policy_targets_path}")
    print(f"Saved value targets to {value_targets_path}")

    return inputs_path, policy_targets_path, value_targets_path