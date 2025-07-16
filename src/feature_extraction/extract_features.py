import chess.pgn
import csv
from pathlib import Path
from multiprocessing import Pool
import argparse

def piece_value(piece):
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    return values[piece.piece_type]

def elo_norm(elo):
    return (elo - 1000) / (3000 - 1000)

def turn_norm(turn):
    return turn / 60

def result_to_numeric(result_str):
    if result_str == "1-0":
        return 2
    elif result_str == "0-1":
        return 0
    elif result_str == "1/2-1/2":
        return 1
    else:
        return None  # IMPORTANT: Aim to dropna your data when training your model

def extract_features(board, white_elo, black_elo, white_castled, black_castled, result):
    features = {}
    features["white_pawns"] = len(board.pieces(chess.PAWN, chess.WHITE))
    features["black_pawns"] = len(board.pieces(chess.PAWN, chess.BLACK))
    features["turn"] = turn_norm(board.fullmove_number)
    features["white_castled"] = white_castled
    features["black_castled"] = black_castled
    features["white_elo"] = elo_norm(white_elo)
    features["black_elo"] = elo_norm(black_elo)
    features["material_diff"] = (
        sum(piece_value(p) for p in board.piece_map().values() if p.color == chess.WHITE) -
        sum(piece_value(p) for p in board.piece_map().values() if p.color == chess.BLACK)
    )
    features["is_white_turn"] = int(board.turn)
    if result is not None:
        features["result"] = result
    return features

def process_games(args):
    pgn_path, start, count, worker_id = args
    results = []
    with open(pgn_path) as f:
        # Skip games until start
        for _ in range(start):
            game = chess.pgn.read_game(f)
            if game is None:
                print(f"Worker {worker_id}: not any more games to skip.")
                return results
        
        for i in range(count):
            game = chess.pgn.read_game(f)
            if game is None:
                print(f"Worker {worker_id}: not any more games to process.")
                break

            if i % 1000 == 0 and i > 0:
                print(f"Worker {worker_id}: procesadas {i} partidas.")

            # Get Elo or 1500 as default
            white_elo = int(game.headers.get("WhiteElo", 1500))
            black_elo = int(game.headers.get("BlackElo", 1500))
            castling_white_done = False
            castling_black_done = False
            board = game.board()

            result_str = game.headers.get("Result", "*")
            result_num = result_to_numeric(result_str)

            for move in game.mainline_moves():
                if board.is_castling(move):
                    if board.turn == chess.WHITE:
                        castling_white_done = True
                    else:
                        castling_black_done = True
                board.push(move)
                features = extract_features(board, white_elo, black_elo, castling_white_done, castling_black_done, result_num)
                results.append(features)
    print(f"Worker {worker_id}: finished, processed positions: {len(results)}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract chess position features from a PGN file in parallel"
    )
    parser.add_argument("--pgn", required=True,
                        help="Path to the input PGN file")
    parser.add_argument("--games", type=int, default=20000,
                        help="Total number of games to process")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--output", default="features.csv",
                        help="Path to the output CSV file")
    args = parser.parse_args()

    chunk = args.games // args.workers
    pool_args = [
        (args.pgn, i * chunk, chunk, i)
        for i in range(args.workers)
    ]

    with Pool(args.workers) as pool:
        result_lists = pool.map(process_games, pool_args)

    all_features = [feat for sub in result_lists for feat in sub]

    fieldnames = [
        "white_pawns", "black_pawns", "turn",
        "white_castled", "black_castled",
        "white_elo", "black_elo",
        "material_diff", "is_white_turn",
        "result"
    ]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for feat in all_features:
            writer.writerow(feat)

    print(f"Done! Wrote {len(all_features)} positions to {args.output}")

if __name__ == "__main__":
    main()