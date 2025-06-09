# test_mcts_utils.py
import chess
from mcts_utils import get_move_index, get_move_from_index, TOTAL_MOVE_CHANNELS
from typing import List # Add this import for type hinting lists

def run_move_mapping_tests():
    print(f"Testing move mapping functions (Total channels: {TOTAL_MOVE_CHANNELS})...")

    # Define test cases as (board_fen, move_uci)
    test_cases: List[tuple[str, str]] = [
        # Standard moves
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4"),
        
        # Castling (needs specific board states)
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"), # White kingside
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"), # White queenside
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"), # Black kingside
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"), # Black queenside

        # White Promotions
        ("8/P7/8/8/8/8/8/8 w - - 0 1", "a7a8q"), # White straight queen promotion
        ("8/P7/8/8/8/8/8/8 w - - 0 1", "a7a8r"), # White straight rook promotion
        ("8/P7/8/8/8/8/8/8 w - - 0 1", "a7a8b"), # White straight bishop promotion
        ("8/P7/8/8/8/8/8/8 w - - 0 1", "a7a8n"), # White straight knight promotion
        ("8/P1k5/8/8/8/8/8/8 w - - 0 1", "a7b8q"), # White diagonal capture queen promotion
        ("8/P1k5/8/8/8/8/8/8 w - - 0 1", "a7b8r"), # White diagonal capture rook promotion

        # Black Promotions
        ("8/8/8/8/8/8/p7/8 b - - 0 1", "a2a1q"), # Black straight queen promotion
        ("8/8/8/8/8/8/p7/8 b - - 0 1", "a2a1r"), # Black straight rook promotion
        ("8/8/8/8/8/8/p7/8 b - - 0 1", "a2a1b"), # Black straight bishop promotion
        ("8/8/8/8/8/8/p7/8 b - - 0 1", "a2a1n"), # Black straight knight promotion
        ("8/8/8/8/8/2K5/p7/8 b - - 0 1", "a2b1q"), # Black diagonal capture queen promotion
        ("8/8/8/8/8/2K5/p7/8 b - - 0 1", "a2b1r"), # Black diagonal capture rook promotion
    ]

    for fen, uci_move_str in test_cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move_str)
        
        # Ensure the move is legal on the current board before testing
        if move not in board.legal_moves:
            print(f"SKIPPED (Illegal Move): {uci_move_str} on board {fen}")
            continue

        idx = get_move_index(board, move)
        if idx is None:
            print(f"FAILED: get_move_index returned None for {move.uci()} on board {board.fen()}")
            continue
        
        if not (0 <= idx < TOTAL_MOVE_CHANNELS):
            print(f"FAILED: Index {idx} out of bounds for {move.uci()}")
            continue

        reconstructed_move = get_move_from_index(idx, board)
        
        if reconstructed_move is None:
            print(f"FAILED: get_move_from_index returned None for index {idx} (original {move.uci()})")
            continue

        # Strict comparison of move objects
        if reconstructed_move != move:
            print(f"FAILED: Original {move.uci()} (promotion={move.promotion}) -> Index {idx} -> Reconstructed {reconstructed_move.uci()} (promotion={reconstructed_move.promotion})")
        else:
            print(f"PASSED: {move.uci()} <-> {idx}")

    print("Move mapping tests complete.")

if __name__ == "__main__":
    run_move_mapping_tests()