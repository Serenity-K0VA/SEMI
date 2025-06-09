# mcts_utils.py (Refined for 4672 moves with fixes)

import chess
import numpy as np
from typing import Union

# Define the 4672-dimensional move channels as commonly seen in AlphaZero-like projects
# This exact mapping is based on common open-source implementations that result in 4672.
# It defines a specific order for all possible moves.

# Define the relative coordinates for 8 queen-like directions (deltas)
# (dx, dy) where dx is file change, dy is rank change
QUEEN_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)
] # N, NE, E, SE, S, SW, W, NW

# Define the relative coordinates for 8 knight moves (deltas)
KNIGHT_DIRECTIONS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

# Order of promotion pieces for consistency (for underpromotions)
PROMOTION_PIECES = [
    chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
]

# --- MAPPING HELPER FUNCTIONS ---

# Helper to get the file and rank from a square
def sq_to_file_rank(sq: chess.Square):
    return chess.square_file(sq), chess.square_rank(sq)

# Helper to get square from file and rank
def file_rank_to_sq(file: int, rank: int):
    return chess.square(file, rank)

def get_move_index(board: chess.Board, move: chess.Move) -> Union[int, None]:
    """
    Converts a chess.Move object to its corresponding index (0-4671).
    This mapping is crucial and must be deterministic and consistent.
    """
    from_sq_file, from_sq_rank = sq_to_file_rank(move.from_square)
    to_sq_file, to_sq_rank = sq_to_file_rank(move.to_square)

    target_square_idx = move.to_square # 0-63

    # Calculate delta file and delta rank
    df = to_sq_file - from_sq_file
    dr = to_sq_rank - from_sq_rank

    # Handle underpromotions (Rook, Bishop, Knight promotions)
    # This also covers queen promotions that are explicitly marked as such (e.g., move.promotion == chess.QUEEN)
    # The AlphaZero paper uses 3 promotion types per direction (R, B, N). Queen promotion is handled implicitly
    # by the straight/diagonal moves if the move leads to a promotion square.
    # However, for strict comparison `chess.Move(..., promotion=chess.QUEEN)` vs `chess.Move(...)`
    # it's better to explicitly map queen promotions here too if the original move has promotion=QUEEN.
    # We maintain the 9 underpromotion channels as in standard AlphaZero.
    # If the move is a pawn move to the last rank, and it is a promotion, it falls into this block.
    if move.promotion is not None: # Check if it's any type of promotion
        if board.piece_type_at(move.from_square) == chess.PAWN and \
           ((board.turn == chess.WHITE and chess.square_rank(move.to_square) == 7) or \
            (board.turn == chess.BLACK and chess.square_rank(move.to_square) == 0)):
            
            # Determine direction of promotion
            if df == 0 and abs(dr) == 1: # Straight push
                prom_dir_idx = 0
            elif df == -1 and abs(dr) == 1: # Diagonal left capture
                prom_dir_idx = 1
            elif df == 1 and abs(dr) == 1: # Diagonal right capture
                prom_dir_idx = 2
            else: return None # Should not happen for valid pawn promotions
            
            # Determine promotion piece type index
            try:
                # Queen is at index 0, Rook at 1, Bishop at 2, Knight at 3.
                # If we map R,B,N to 0,1,2 for channels, then Queen needs careful handling.
                # Standard AlphaZero has 3 underpromotion types. Queen promotion typically falls into the 56 queen-like channels.
                # To maintain `move == reconstructed_move` for `promotion=chess.QUEEN` object:
                # We should *not* explicitly encode queen promotions here if 4672-D is based on 9 underpromotion types only.
                # Instead, `get_move_from_index` should infer queen promotions.
                # So, this block should ONLY handle non-queen promotions (underpromotions).
                if move.promotion == chess.QUEEN:
                    # If it's a queen promotion, let it fall through to the queen-like move handling
                    # This implies get_move_from_index needs to infer queen promotion if it's a pawn move to 8th/1st rank
                    pass
                else:
                    # For underpromotions (Rook, Bishop, Knight)
                    prom_type_idx = PROMOTION_PIECES.index(move.promotion) - 1 # R=0, B=1, N=2
                    channel_idx = 64 + (prom_dir_idx * 3) + prom_type_idx
                    return target_square_idx * 73 + channel_idx
            except ValueError:
                # Should not happen if PROMOTION_PIECES list is correct
                return None


    # Handle Knight moves
    if abs(df * dr) == 2 and abs(df) + abs(dr) == 3: # Check for L-shape
        for i, (kdf, kdr) in enumerate(KNIGHT_DIRECTIONS):
            if df == kdf and dr == kdr:
                channel_idx = 56 + i # Knight channels 56-63
                return target_square_idx * 73 + channel_idx
        return None

    # Handle Queen-like (Rook, Bishop, Queen, and implicit Queen promotions) moves
    # This also covers straight pawn pushes that are queen promotions (e.g. e7e8q)
    if df == 0 or dr == 0 or abs(df) == abs(dr):
        if df == 0: # Vertical (N/S)
            direction_idx = 0 if dr > 0 else 4 # N or S
        elif dr == 0: # Horizontal (E/W)
            direction_idx = 2 if df > 0 else 6 # E or W
        elif df > 0 and dr > 0: direction_idx = 1 # NE
        elif df > 0 and dr < 0: direction_idx = 3 # SE
        elif df < 0 and dr < 0: direction_idx = 5 # SW
        elif df < 0 and dr > 0: direction_idx = 7 # NW
        else: return None # Should not happen for valid queen-like moves

        steps_idx = max(abs(df), abs(dr)) - 1 # Steps are 0-6 (for 1-7 squares away)

        channel_idx = (direction_idx * 7) + steps_idx # Queen-like channels 0-55
        return target_square_idx * 73 + channel_idx
    
    return None # If move doesn't fit any defined pattern

def get_move_from_index(index: int, board: chess.Board) -> Union[chess.Move, None]:
    """
    Converts an index (0-4671) back to a chess.Move object given the current board state.
    This is trickier because `from_square` is not directly encoded for some types.
    """
    if not (0 <= index < 4672):
        return None

    target_square_idx = index // 73
    channel_idx = index % 73

    # Reconstruct move based on channel_idx
    if channel_idx < 56: # Queen-like moves (0-55)
        direction_idx = channel_idx // 7
        steps_idx = channel_idx % 7
        steps = steps_idx + 1

        ddf, ddr = QUEEN_DIRECTIONS[direction_idx]
        from_sq_file = chess.square_file(target_square_idx) - ddf * steps
        from_sq_rank = chess.square_rank(target_square_idx) - ddr * steps
        
        if not (0 <= from_sq_file < 8 and 0 <= from_sq_rank < 8):
            return None
        
        from_sq = file_rank_to_sq(from_sq_file, from_sq_rank)
        
        # === LOGIC FOR QUEEN PROMOTION INFERENCE ===
        # If it's a pawn moving to the final rank for promotion, infer it's a queen promotion
        # (since underpromotions are handled in their own channels)
        piece_on_from_sq = board.piece_type_at(from_sq)
        is_promotion_rank = (board.turn == chess.WHITE and chess.square_rank(target_square_idx) == 7) or \
                            (board.turn == chess.BLACK and chess.square_rank(target_square_idx) == 0)

        if piece_on_from_sq == chess.PAWN and is_promotion_rank:
            move = chess.Move(from_sq, target_square_idx, promotion=chess.QUEEN)
            return move
        # === END LOGIC ===

        move = chess.Move(from_sq, target_square_idx)
        return move

    elif channel_idx < 64: # Knight moves (56-63)
        knight_direction_idx = channel_idx - 56
        kdf, kdr = KNIGHT_DIRECTIONS[knight_direction_idx]

        from_sq_file = chess.square_file(target_square_idx) - kdf
        from_sq_rank = chess.square_rank(target_square_idx) - kdr
        
        if not (0 <= from_sq_file < 8 and 0 <= from_sq_rank < 8):
            return None
        
        from_sq = file_rank_to_sq(from_sq_file, from_sq_rank)
        move = chess.Move(from_sq, target_square_idx)
        return move

    elif channel_idx < 73: # Underpromotions (64-72)
        prom_channel_offset = channel_idx - 64
        prom_dir_idx = prom_channel_offset // 3
        prom_type_idx = prom_channel_offset % 3
        
        promotion_piece = PROMOTION_PIECES[prom_type_idx + 1] # +1 because PROMOTION_PIECES starts with QUEEN

        if board.turn == chess.WHITE:
            if prom_dir_idx == 0: # Straight push
                from_sq = target_square_idx - 8
            elif prom_dir_idx == 1: # Diagonal left capture
                from_sq = target_square_idx + 1 - 8
            elif prom_dir_idx == 2: # Diagonal right capture
                from_sq = target_square_idx - 1 - 8
            else: return None
        else: # Black's turn
            if prom_dir_idx == 0: # Straight push
                from_sq = target_square_idx + 8
            elif prom_dir_idx == 1: # Diagonal left capture
                from_sq = target_square_idx - 1 + 8
            elif prom_dir_idx == 2: # Diagonal right capture
                from_sq = target_square_idx + 1 + 8
            else: return None

        if not (0 <= from_sq < 64): # Ensure from_sq is within board bounds
            return None

        move = chess.Move(from_sq, target_square_idx, promotion=promotion_piece)
        return move
    
    return None

def decode_policy_output(policy_output_array: np.ndarray, board: chess.Board) -> dict:
    """
    Converts the 4672-dimensional policy output array into a dictionary
    mapping legal UCI moves to their predicted probabilities.
    """
    move_probs = {}
    
    legal_moves_map = {move.uci(): move for move in board.legal_moves}
    
    for i, prob in enumerate(policy_output_array):
        if prob < 0: prob = 0 # Ensure probabilities are non-negative
        
        move = get_move_from_index(i, board)
        if move and move.uci() in legal_moves_map:
            move_probs[move.uci()] = prob
            
    total_prob = sum(move_probs.values())
    if total_prob > 0:
        for uci_move in move_probs:
            move_probs[uci_move] /= total_prob # Normalize probabilities
    else:
        # Fallback: if no legal moves get a probability, distribute uniformly
        legal_moves = list(board.legal_moves)
        if legal_moves:
            for move in legal_moves:
                move_probs[move.uci()] = 1.0 / len(legal_moves)
        # If no legal moves at all (e.g., game over), move_probs remains empty.
    
    return move_probs

def encode_policy_target(policy_target_map: dict, board: chess.Board, policy_output_size=4672) -> np.ndarray:
    """
    Converts a dictionary of {uci_move: probability} into the 4672-dimensional
    probability distribution array that your policy head expects.
    """
    encoded_policy = np.zeros(policy_output_size, dtype=np.float32)
    
    for uci_move, prob in policy_target_map.items():
        move = chess.Move.from_uci(uci_move)
        idx = get_move_index(board, move)
        if idx is not None:
            encoded_policy[idx] = prob
        else:
            # This warning is useful for debugging if a move in the target map
            # cannot be converted to an index by your mapping.
            print(f"Warning: Could not encode move {uci_move} to index.")
            
    return encoded_policy

TOTAL_MOVE_CHANNELS = 73 * 64 # This correctly calculates 4672