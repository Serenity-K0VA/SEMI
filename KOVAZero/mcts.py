# mcts.py
from NNKOVA3 import board_to_tensor
import chess
import numpy as np
import math
from mcts_utils import ( # NEW/UPDATED IMPORTS
    get_move_index, get_move_from_index, decode_policy_output, encode_policy_target,
    QUEEN_DIRECTIONS, KNIGHT_DIRECTIONS, PROMOTION_PIECES, TOTAL_MOVE_CHANNELS
)
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None, prior_p=0.0):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = {}  # Map from legal move to MCTSNode object
        self.N = 0  # Visit count
        self.W = 0.0 # Total action value (sum of values propagated back)
        self.Q = 0.0 # Average action value (Q = W / N)
        self.P = prior_p # Prior probability from neural network

        self.is_expanded = False # Flag to track if children have been added

    def ucb_score(self, parent_N, c_puct=1.0):
        """Calculate the UCB score for selection."""
        if self.N == 0:
            return self.P # Only exploration term initially
        return self.Q + c_puct * self.P * (math.sqrt(parent_N) / (1 + self.N))

    def expand(self, policy_probs, value_estimate, legal_moves):
        """Expand the node and initialize children."""
        self.is_expanded = True
        for move in legal_moves:
            if move.uci() in policy_probs: # Check if move is in NN policy output
                prior = policy_probs[move.uci()]
            else:
                prior = 0.0 # Assign zero prior to illegal/uncovered moves
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move, prior_p=prior)

        # Apply a small amount of noise to the root policy to encourage exploration
        # (Dirichlet noise, typical in AlphaZero, can be added here)
        if self.parent is None: # This is the root node
             # Placeholder for Dirichlet noise if you implement it later
             # policy_probs = self.add_dirichlet_noise(policy_probs)
             pass

    def backpropagate(self, value):
        """Update node statistics from a simulation result."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backpropagate(value)

# === MCTS Search Function ===
def mcts_search(initial_board: chess.Board, model, num_simulations: int, c_puct=1.0, temperature=1.0):
    root = MCTSNode(initial_board.copy())

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # 1. Selection
        while node.is_expanded and node.children: # Ensure node has children before trying to select
            legal_moves = list(node.children.keys())
            if not legal_moves: # Should not happen if expansion is done correctly on non-terminal nodes
                break

            # Find best child based on UCB score
            best_move = None
            best_score = -float('inf')

            for move in legal_moves:
                child_node = node.children[move]
                score = child_node.ucb_score(node.N, c_puct)
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move is None: # Fallback if no best move found (e.g., if all scores are -inf or no legal moves)
                break
            
            node = node.children[best_move]
            search_path.append(node)

        # Check if current node is a terminal node before expansion
        if node.board.is_game_over():
            value = get_game_result_value(node.board)
            node.backpropagate(value)
            continue # Go to next simulation

        # 2. Expansion
        # Query the neural network for policy (P) and value (V)
        # Note: You need to implement board_to_tensor and your model.predict
        # and parse policy output into a dictionary of {uci_move: prob}
        board_tensor = board_to_tensor(node.board)
        # model.predict expects a batch, so reshape
        policy_logits, value_raw = model.predict(np.expand_dims(board_tensor, axis=0), verbose=0)

        # Process policy logits into a dictionary of {uci_move: prob}
        # You'll need to map the 4672 policy output values back to chess moves
        policy_probs_dict = decode_policy_output(policy_logits[0], node.board) # Implement this helper

        value_estimate = value_raw[0][0] # Get scalar value

        legal_moves = list(node.board.legal_moves)
        node.expand(policy_probs_dict, value_estimate, legal_moves)

        # 3. Evaluation (Value from NN)
        # The value estimate from the NN is used directly for backprop
        value = value_estimate # This is the immediate value, will be updated during backprop

        # 4. Backpropagation
        node.backpropagate(value)

    # After simulations, determine the best move and policy targets
    # Policy targets are based on visit counts N(s,a) raised to a temperature
    visit_counts = {move: child.N for move, child in root.children.items()}
    
    # Apply temperature for exploration during training vs. exploitation during play
    if temperature == 0: # For actual play, pick the most visited move
        best_move = max(visit_counts, key=visit_counts.get)
        policy_target_map = {move.uci(): (1.0 if move == best_move else 0.0) for move in root.children.keys()}
    else: # For self-play training, sample moves based on N(s,a)^(1/temp)
        # Calculate raw probabilities based on visit counts
        sum_visits = sum(v ** (1/temperature) for v in visit_counts.values())
        policy_target_map = {move.uci(): (count ** (1/temperature)) / sum_visits for move, count in visit_counts.items()}
        
    return policy_target_map # This is your policy target, a dictionary of {uci_move: probability}


def get_game_result_value(board: chess.Board):
    """
    Returns 1.0 for White win, -1.0 for Black win, 0.0 for draw.
    Assumes game is over.
    """
    if board.is_checkmate():
        return 1.0 if board.outcome().winner == chess.WHITE else -1.0
    return 0.0 # Draw or stalemate

# You need to implement these helpers to work with your model's policy output
def decode_policy_output(policy_output_array, board: chess.Board):
    """
    Converts the 4672-dimensional policy output array into a dictionary
    mapping legal UCI moves to their predicted probabilities.
    This is complex and requires a fixed mapping from index to move.
    """
    # Placeholder: You need to implement your mapping logic here.
    # For example, if you have a pre-defined list of all possible 4672 moves:
    # all_possible_moves = [...] # Your mapping for 4672 moves
    # move_probs = {all_possible_moves[i]: policy_output_array[i] for i in range(len(all_possible_moves))}
    # Filter for legal moves
    
    # For now, a simplified dummy structure to show interaction:
    legal_moves = list(board.legal_moves)
    dummy_probs = {move.uci(): 1.0 / len(legal_moves) for move in legal_moves} # Uniform for now
    
    # Important: You'll need a precise mapping for the 4672 output
    # This involves defining an order for all possible moves (including promotions, castling etc)
    # and then mapping the 4672 elements of policy_output_array to these moves.
    # For example, mapping 0-63 to a1-h8, then 64-127 for pawn moves, etc.
    # This part requires careful engineering to match your policy head's output structure.
    return dummy_probs

def encode_policy_target(policy_target_map, board: chess.Board, policy_output_size=4672):
    """
    Converts a dictionary of {uci_move: probability} into the 4672-dimensional
    one-hot or probability distribution array that your policy head expects.
    This also requires the same fixed mapping as decode_policy_output.
    """
    encoded_policy = np.zeros(policy_output_size, dtype=np.float32)
    # Placeholder: You need to implement your mapping logic here
    # Example:
    # for move_uci, prob in policy_target_map.items():
    #     idx = get_index_for_move_uci(move_uci) # Your function to map UCI to index
    #     if idx is not None:
    #         encoded_policy[idx] = prob
    
    # For now, a dummy implementation where it's all zeros if not precisely mapped
    # This WILL NOT WORK FOR TRAINING until you implement the real encoding.
    return encoded_policy