"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import mcp.types as types


import io
import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import chess
import chess.pgn
import chess.engine


mcp = FastMCP("Echo Server", port=3000, stateless_http=True, debug=True)


# ---------------------------
# Data Models
# ---------------------------


@dataclass
class CoachNote:
    move_no: int
    side: str
    tag: str  # "Blunder", "Mistake", "Inaccuracy"
    delta_pawns: float
    played_san: str
    best_san: str
    best_line: str
    comment: str


@dataclass
class MistakePattern:
    pattern_type: str
    frequency: int
    total_delta: float
    avg_delta: float
    examples: List[str]
    description: str


# ---------------------------
# Chess Analysis Core
# ---------------------------

BASE = "https://api.chess.com/pub"
SCORE_MATE = 10000  # map mate to large cp value for consistency


def _get_archives(username: str) -> List[str]:
    url = f"{BASE}/player/{username}/games/archives"
    r = requests.get(url, headers={"User-Agent": "python-chess-data/1.0"})
    r.raise_for_status()
    return r.json()["archives"]


def _get_games_json(archive_url: str) -> dict:
    r = requests.get(archive_url, headers={"User-Agent": "python-chess-data/1.0"})
    r.raise_for_status()
    return r.json()


def get_latest_game_pgn(username: str) -> str:
    """
    Returns PGN text of the most recent finished game.
    Walks archives from newest backwards until a non-empty month is found.
    """
    archives = _get_archives(username)
    for archive_url in reversed(
        archives
    ):  # most recent at the end; iterate newest -> oldest
        data = _get_games_json(archive_url)
        games = data.get("games", [])
        if not games:
            continue
        # some months include live, daily, variants; just take the last finished listed
        last_game = games[-1]
        pgn = last_game.get("pgn")
        if pgn:
            return pgn
    raise ValueError(f"No finished games found for user '{username}'")


def get_games_last_30_days(username: str) -> List[str]:
    """
    Returns list of PGN texts for all finished games in the last 30 days.
    """
    thirty_days_ago = datetime.now() - timedelta(days=30)
    all_pgns = []

    archives = _get_archives(username)
    for archive_url in reversed(archives):  # most recent first
        # Extract date from archive URL (format: YYYY/MM)
        try:
            date_part = archive_url.split("/")[-2] + "/" + archive_url.split("/")[-1]
            archive_date = datetime.strptime(date_part, "%Y/%m")

            # Skip if archive is older than 30 days
            if archive_date < thirty_days_ago.replace(day=1):
                continue

        except (ValueError, IndexError):
            continue

        data = _get_games_json(archive_url)
        games = data.get("games", [])

        for game in games:
            pgn = game.get("pgn")
            if pgn:
                # Check if game is within last 30 days
                try:
                    # Parse game date from PGN headers
                    game_headers = chess.pgn.read_headers(io.StringIO(pgn))
                    if game_headers and "Date" in game_headers:
                        game_date = datetime.strptime(game_headers["Date"], "%Y.%m.%d")
                        if game_date >= thirty_days_ago:
                            all_pgns.append(pgn)
                except (ValueError, TypeError):
                    # If we can't parse date, include it (better to include than miss)
                    all_pgns.append(pgn)

    return all_pgns


def _cp(score: chess.engine.PovScore) -> int:
    val = score.score(mate_score=SCORE_MATE)
    return 0 if val is None else int(val)


def _format_pv(
    board: chess.Board, pv: List[chess.Move], max_len: int = 6
) -> Tuple[str, str]:
    b = board.copy()
    sans = []
    for m in pv[:max_len]:
        sans.append(b.san(m))
        b.push(m)
    best_san = sans[0] if sans else "—"
    return best_san, " ".join(sans)


def analyze_pgn_with_stockfish(
    pgn_text: str,
    engine_path: str,
    depth: int = 16,
    multipv: int = 1,
    thresholds=(50, 150, 300),  # (Inaccuracy, Mistake, Blunder) in centipawns
) -> List[CoachNote]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        raise ValueError("Could not parse PGN")

    ina_t, mis_t, blu_t = thresholds
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    notes: List[CoachNote] = []

    try:
        board = game.board()
        for ply_idx, move in enumerate(game.mainline_moves(), start=1):
            mover_color = board.turn
            mover_name = "White" if mover_color == chess.WHITE else "Black"
            move_no = (ply_idx + 1) // 2
            played_san = board.san(move)

            # Evaluate BEFORE (mover POV)
            info_before = engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=multipv
            )
            if isinstance(info_before, list):  # multipv>1 returns list
                info_before = info_before[0]
            cp_before = _cp(info_before["score"].pov(mover_color))
            pv_before = info_before.get("pv", [])
            best_san, best_line = _format_pv(board, pv_before)

            # Play move, evaluate AFTER (still mover POV)
            board.push(move)
            info_after = engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=1
            )
            if isinstance(info_after, list):  # multipv>1 returns list
                info_after = info_after[0]
            cp_after = _cp(info_after["score"].pov(mover_color))

            delta_cp = cp_before - cp_after  # positive = worse for mover
            tag = None
            if delta_cp >= blu_t:
                tag = "Blunder"
            elif delta_cp >= mis_t:
                tag = "Mistake"
            elif delta_cp >= ina_t:
                tag = "Inaccuracy"

            if tag:
                comment_bits = []
                last = board.peek()

                # Analyze the specific move and position
                if last:
                    piece_type = board.piece_type_at(last.to_square)
                    from_square = last.from_square
                    to_square = last.to_square

                    # Pawn-specific analysis
                    if piece_type == chess.PAWN:
                        if delta_cp >= mis_t:
                            if "x" in played_san:  # Pawn capture
                                comment_bits.append(
                                    f"You captured with the pawn, but this weakened your pawn structure. The engine suggests {best_san} instead."
                                )
                            else:  # Pawn push
                                comment_bits.append(
                                    f"This pawn push overextended your position and weakened key squares. Consider {best_san} to maintain better structure."
                                )
                        else:
                            comment_bits.append(
                                f"Pawn moves require careful consideration of structure. {best_san} would have been more solid."
                            )

                    # King safety analysis
                    elif piece_type == chess.KING:
                        if delta_cp >= mis_t:
                            comment_bits.append(
                                f"Moving the king exposed it to danger. {best_san} would have been safer."
                            )
                        else:
                            comment_bits.append(
                                f"King safety is crucial. {best_san} would have been more prudent."
                            )

                    # Queen analysis
                    elif piece_type == chess.QUEEN:
                        if ply_idx <= 12 and delta_cp >= mis_t:
                            comment_bits.append(
                                f"Early queen moves often lose tempi and expose the queen to attack. {best_san} would develop a minor piece first."
                            )
                        elif delta_cp >= mis_t:
                            comment_bits.append(
                                f"The queen move was too aggressive and exposed it to threats. {best_san} would have been more positional."
                            )
                        else:
                            comment_bits.append(
                                f"Queen moves need careful timing. {best_san} would have been more accurate."
                            )

                    # Minor pieces analysis
                    elif piece_type in [chess.KNIGHT, chess.BISHOP]:
                        if delta_cp >= mis_t:
                            comment_bits.append(
                                f"This {chess.piece_name(piece_type).lower()} move was inaccurate and lost material/position. {best_san} would have been better."
                            )
                        else:
                            comment_bits.append(
                                f"The {chess.piece_name(piece_type).lower()} move wasn't optimal. {best_san} would have been more precise."
                            )

                    # Rook analysis
                    elif piece_type == chess.ROOK:
                        if delta_cp >= mis_t:
                            comment_bits.append(
                                f"The rook move was premature and didn't improve your position. {best_san} would have been more effective."
                            )
                        else:
                            comment_bits.append(
                                f"Rook moves should be well-timed. {best_san} would have been more accurate."
                            )

                # General positional advice based on delta
                if delta_cp >= 300:  # Blunder
                    comment_bits.append(
                        f"This was a serious blunder that lost significant advantage. The engine's line {best_line} shows the correct continuation."
                    )
                elif delta_cp >= 150:  # Mistake
                    comment_bits.append(
                        f"This mistake gave your opponent a clear advantage. Study the engine's suggestion: {best_line}"
                    )
                else:  # Inaccuracy
                    comment_bits.append(
                        f"This inaccuracy slightly worsened your position. {best_san} would have maintained equality."
                    )

                # Add general advice if no specific comments
                if not comment_bits:
                    comment_bits.append(
                        f"Consider the engine's suggested line: {best_line}"
                    )

                comment = " ".join(comment_bits)

                notes.append(
                    CoachNote(
                        move_no=move_no,
                        side=mover_name,
                        tag=tag,
                        delta_pawns=round(delta_cp / 100.0, 2),
                        played_san=played_san,
                        best_san=best_san,
                        best_line=best_line,
                        comment=comment,
                    )
                )
        return notes
    finally:
        engine.quit()


# ---------------------------
# Pattern Analysis Functions
# ---------------------------


def _analyze_tactical_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze tactical mistakes and piece losses."""
    patterns = []

    # Group by piece type and mistake severity
    piece_losses = defaultdict(lambda: defaultdict(list))
    for note in notes:
        if note.delta_pawns >= 1.5:  # Significant material loss
            move_san = note.played_san
            if move_san[0].isupper():  # Piece move
                piece_type = move_san[0]
                piece_losses[piece_type][note.tag].append(note)

    # Analyze each piece type
    for piece_type, mistakes_by_severity in piece_losses.items():
        total_mistakes = sum(
            len(mistakes) for mistakes in mistakes_by_severity.values()
        )
        if total_mistakes >= 3:  # Only report if significant pattern
            blunders = mistakes_by_severity.get("Blunder", [])
            mistakes = mistakes_by_severity.get("Mistake", [])

            # Analyze specific patterns for each piece
            if piece_type == "N":  # Knight
                patterns.append(
                    MistakePattern(
                        pattern_type=f"Knight losses",
                        frequency=total_mistakes,
                        total_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ),
                        avg_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        )
                        / total_mistakes,
                        examples=[
                            m.played_san
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ][:5],
                        description=f"You lose your knight {total_mistakes} times, often by moving it to squares where it can be trapped by pawns or pinned pieces.",
                    )
                )
            elif piece_type == "Q":  # Queen
                early_queen = [
                    m
                    for mistakes in mistakes_by_severity.values()
                    for m in mistakes
                    if m.move_no <= 15
                ]
                if len(early_queen) >= 2:
                    patterns.append(
                        MistakePattern(
                            pattern_type=f"Early Queen losses",
                            frequency=len(early_queen),
                            total_delta=sum(m.delta_pawns for m in early_queen),
                            avg_delta=sum(m.delta_pawns for m in early_queen)
                            / len(early_queen),
                            examples=[m.played_san for m in early_queen][:3],
                            description=f"You lose your queen {len(early_queen)} times in the opening by moving it too early, allowing opponent to develop with tempo.",
                        )
                    )
            elif piece_type == "B":  # Bishop
                patterns.append(
                    MistakePattern(
                        pattern_type=f"Bishop losses",
                        frequency=total_mistakes,
                        total_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ),
                        avg_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        )
                        / total_mistakes,
                        examples=[
                            m.played_san
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ][:5],
                        description=f"You lose your bishop {total_mistakes} times, often by pinning it to your king without an escape square or moving it to trapped squares.",
                    )
                )

    return patterns


def _analyze_positional_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze positional mistakes."""
    patterns = []

    # King safety issues
    king_safety_mistakes = [
        n
        for n in notes
        if "O-O" in n.played_san or "O-O-O" in n.played_san or "K" in n.played_san
    ]
    if len(king_safety_mistakes) >= 3:
        patterns.append(
            MistakePattern(
                pattern_type="King safety issues",
                frequency=len(king_safety_mistakes),
                total_delta=sum(m.delta_pawns for m in king_safety_mistakes),
                avg_delta=sum(m.delta_pawns for m in king_safety_mistakes)
                / len(king_safety_mistakes),
                examples=[m.played_san for m in king_safety_mistakes][:5],
                description=f"You make {len(king_safety_mistakes)} king safety mistakes, often by moving the f-pawn without castling first or exposing your king to attacks.",
            )
        )

    # Pawn structure issues
    pawn_mistakes = [
        n for n in notes if n.played_san[0].islower() and n.delta_pawns >= 0.5
    ]
    if len(pawn_mistakes) >= 5:
        patterns.append(
            MistakePattern(
                pattern_type="Pawn structure mistakes",
                frequency=len(pawn_mistakes),
                total_delta=sum(m.delta_pawns for m in pawn_mistakes),
                avg_delta=sum(m.delta_pawns for m in pawn_mistakes)
                / len(pawn_mistakes),
                examples=[m.played_san for m in pawn_mistakes][:5],
                description=f"You make {len(pawn_mistakes)} pawn structure mistakes, often by creating isolated pawns, doubled pawns, or weakening key squares.",
            )
        )

    return patterns


def _analyze_opening_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze opening-specific mistakes."""
    patterns = []

    opening_mistakes = [n for n in notes if n.move_no <= 15]
    if len(opening_mistakes) >= 5:
        # Early queen moves
        early_queen = [n for n in opening_mistakes if "Q" in n.played_san]
        if len(early_queen) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Early Queen moves",
                    frequency=len(early_queen),
                    total_delta=sum(m.delta_pawns for m in early_queen),
                    avg_delta=sum(m.delta_pawns for m in early_queen)
                    / len(early_queen),
                    examples=[m.played_san for m in early_queen][:3],
                    description=f"You move your queen too early {len(early_queen)} times in the opening, losing tempi and exposing it to attacks.",
                )
            )

        # Premature attacks
        aggressive_moves = [
            n
            for n in opening_mistakes
            if n.delta_pawns >= 1.0 and ("x" in n.played_san or "+" in n.played_san)
        ]
        if len(aggressive_moves) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Premature attacks",
                    frequency=len(aggressive_moves),
                    total_delta=sum(m.delta_pawns for m in aggressive_moves),
                    avg_delta=sum(m.delta_pawns for m in aggressive_moves)
                    / len(aggressive_moves),
                    examples=[m.played_san for m in aggressive_moves][:3],
                    description=f"You make {len(aggressive_moves)} premature attacks in the opening before completing development, weakening your position.",
                )
            )

    return patterns


def _analyze_endgame_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze endgame-specific mistakes."""
    patterns = []

    endgame_mistakes = [n for n in notes if n.move_no >= 30]
    if len(endgame_mistakes) >= 5:
        # King activity in endgame
        king_moves = [n for n in endgame_mistakes if "K" in n.played_san]
        if len(king_moves) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Endgame King activity",
                    frequency=len(king_moves),
                    total_delta=sum(m.delta_pawns for m in king_moves),
                    avg_delta=sum(m.delta_pawns for m in king_moves) / len(king_moves),
                    examples=[m.played_san for m in king_moves][:3],
                    description=f"You make {len(king_moves)} king activity mistakes in the endgame, often by not following the rule of the square or not centralizing your king.",
                )
            )

        # Pawn endgame mistakes
        pawn_endgame = [
            n
            for n in endgame_mistakes
            if n.played_san[0].islower() and n.delta_pawns >= 1.0
        ]
        if len(pawn_endgame) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Pawn endgame mistakes",
                    frequency=len(pawn_endgame),
                    total_delta=sum(m.delta_pawns for m in pawn_endgame),
                    avg_delta=sum(m.delta_pawns for m in pawn_endgame)
                    / len(pawn_endgame),
                    examples=[m.played_san for m in pawn_endgame][:3],
                    description=f"You make {len(pawn_endgame)} pawn endgame mistakes, often by not calculating pawn races correctly or not using opposition.",
                )
            )

    return patterns


def _analyze_calculation_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze calculation and tactical mistakes."""
    patterns = []

    # Missed tactics
    tactical_mistakes = [
        n
        for n in notes
        if n.delta_pawns >= 2.0 and ("x" in n.played_san or "+" in n.played_san)
    ]
    if len(tactical_mistakes) >= 3:
        patterns.append(
            MistakePattern(
                pattern_type="Missed tactics",
                frequency=len(tactical_mistakes),
                total_delta=sum(m.delta_pawns for m in tactical_mistakes),
                avg_delta=sum(m.delta_pawns for m in tactical_mistakes)
                / len(tactical_mistakes),
                examples=[m.played_san for m in tactical_mistakes][:3],
                description=f"You miss {len(tactical_mistakes)} simple tactics because you don't check for checks, captures, and threats systematically.",
            )
        )

    # Back-rank issues
    back_rank_mistakes = [
        n
        for n in notes
        if "R" in n.played_san and n.move_no >= 20 and n.delta_pawns >= 1.0
    ]
    if len(back_rank_mistakes) >= 2:
        patterns.append(
            MistakePattern(
                pattern_type="Back-rank weaknesses",
                frequency=len(back_rank_mistakes),
                total_delta=sum(m.delta_pawns for m in back_rank_mistakes),
                avg_delta=sum(m.delta_pawns for m in back_rank_mistakes)
                / len(back_rank_mistakes),
                examples=[m.played_san for m in back_rank_mistakes][:3],
                description=f"You create {len(back_rank_mistakes)} back-rank weaknesses by moving your rook without preparing an escape square for your king.",
            )
        )

    return patterns


def _identify_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """
    Identify specific behavioral patterns from a list of coaching notes.
    """
    patterns = []

    # Group by mistake type and analyze specific behaviors
    by_type = defaultdict(list)
    for note in notes:
        by_type[note.tag].append(note)

    # Analyze specific behavioral patterns
    patterns.extend(_analyze_tactical_patterns(notes))
    patterns.extend(_analyze_positional_patterns(notes))
    patterns.extend(_analyze_opening_patterns(notes))
    patterns.extend(_analyze_endgame_patterns(notes))
    patterns.extend(_analyze_calculation_patterns(notes))

    # Sort by frequency and impact
    patterns.sort(key=lambda p: (p.frequency, p.avg_delta), reverse=True)
    return patterns[:15]  # Return top 15 patterns


# ---------------------------
# MCP Server Setup
# ---------------------------

mcp = FastMCP("Chess Analysis Server")



# ---------------------------
# Data Models
# ---------------------------


@dataclass
class CoachNote:
    move_no: int
    side: str
    tag: str  # "Blunder", "Mistake", "Inaccuracy"
    delta_pawns: float
    played_san: str
    best_san: str
    best_line: str
    comment: str


@dataclass
class MistakePattern:
    pattern_type: str
    frequency: int
    total_delta: float
    avg_delta: float
    examples: List[str]
    description: str


# ---------------------------
# Chess Analysis Core
# ---------------------------

BASE = "https://api.chess.com/pub"
SCORE_MATE = 10000  # map mate to large cp value for consistency


def _get_archives(username: str) -> List[str]:
    url = f"{BASE}/player/{username}/games/archives"
    r = requests.get(url, headers={"User-Agent": "python-chess-data/1.0"})
    r.raise_for_status()
    return r.json()["archives"]


def _get_games_json(archive_url: str) -> dict:
    r = requests.get(archive_url, headers={"User-Agent": "python-chess-data/1.0"})
    r.raise_for_status()
    return r.json()


def get_latest_game_pgn(username: str) -> str:
    """
    Returns PGN text of the most recent finished game.
    Walks archives from newest backwards until a non-empty month is found.
    """
    archives = _get_archives(username)
    for archive_url in reversed(
        archives
    ):  # most recent at the end; iterate newest -> oldest
        data = _get_games_json(archive_url)
        games = data.get("games", [])
        if not games:
            continue
        # some months include live, daily, variants; just take the last finished listed
        last_game = games[-1]
        pgn = last_game.get("pgn")
        if pgn:
            return pgn
    raise ValueError(f"No finished games found for user '{username}'")


def get_games_last_30_days(username: str) -> List[str]:
    """
    Returns list of PGN texts for all finished games in the last 30 days.
    """
    thirty_days_ago = datetime.now() - timedelta(days=30)
    all_pgns = []

    archives = _get_archives(username)
    for archive_url in reversed(archives):  # most recent first
        # Extract date from archive URL (format: YYYY/MM)
        try:
            date_part = archive_url.split("/")[-2] + "/" + archive_url.split("/")[-1]
            archive_date = datetime.strptime(date_part, "%Y/%m")

            # Skip if archive is older than 30 days
            if archive_date < thirty_days_ago.replace(day=1):
                continue

        except (ValueError, IndexError):
            continue

        data = _get_games_json(archive_url)
        games = data.get("games", [])

        for game in games:
            pgn = game.get("pgn")
            if pgn:
                # Check if game is within last 30 days
                try:
                    # Parse game date from PGN headers
                    game_headers = chess.pgn.read_headers(io.StringIO(pgn))
                    if game_headers and "Date" in game_headers:
                        game_date = datetime.strptime(game_headers["Date"], "%Y.%m.%d")
                        if game_date >= thirty_days_ago:
                            all_pgns.append(pgn)
                except (ValueError, TypeError):
                    # If we can't parse date, include it (better to include than miss)
                    all_pgns.append(pgn)

    return all_pgns


def _cp(score: chess.engine.PovScore) -> int:
    val = score.score(mate_score=SCORE_MATE)
    return 0 if val is None else int(val)


def _format_pv(
    board: chess.Board, pv: List[chess.Move], max_len: int = 6
) -> Tuple[str, str]:
    b = board.copy()
    sans = []
    for m in pv[:max_len]:
        sans.append(b.san(m))
        b.push(m)
    best_san = sans[0] if sans else "—"
    return best_san, " ".join(sans)


def analyze_pgn_with_stockfish(
    pgn_text: str,
    engine_path: str,
    depth: int = 16,
    multipv: int = 1,
    thresholds=(50, 150, 300),  # (Inaccuracy, Mistake, Blunder) in centipawns
) -> List[CoachNote]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        raise ValueError("Could not parse PGN")

    ina_t, mis_t, blu_t = thresholds
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    notes: List[CoachNote] = []

    try:
        board = game.board()
        for ply_idx, move in enumerate(game.mainline_moves(), start=1):
            mover_color = board.turn
            mover_name = "White" if mover_color == chess.WHITE else "Black"
            move_no = (ply_idx + 1) // 2
            played_san = board.san(move)

            # Evaluate BEFORE (mover POV)
            info_before = engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=multipv
            )
            if isinstance(info_before, list):  # multipv>1 returns list
                info_before = info_before[0]
            cp_before = _cp(info_before["score"].pov(mover_color))
            pv_before = info_before.get("pv", [])
            best_san, best_line = _format_pv(board, pv_before)

            # Play move, evaluate AFTER (still mover POV)
            board.push(move)
            info_after = engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=1
            )
            if isinstance(info_after, list):  # multipv>1 returns list
                info_after = info_after[0]
            cp_after = _cp(info_after["score"].pov(mover_color))

            delta_cp = cp_before - cp_after  # positive = worse for mover
            tag = None
            if delta_cp >= blu_t:
                tag = "Blunder"
            elif delta_cp >= mis_t:
                tag = "Mistake"
            elif delta_cp >= ina_t:
                tag = "Inaccuracy"

            if tag:
                comment_bits = []
                last = board.peek()

                # Analyze the specific move and position
                if last:
                    piece_type = board.piece_type_at(last.to_square)
                    from_square = last.from_square
                    to_square = last.to_square

                    # Pawn-specific analysis
                    if piece_type == chess.PAWN:
                        if delta_cp >= mis_t:
                            if "x" in played_san:  # Pawn capture
                                comment_bits.append(
                                    f"You captured with the pawn, but this weakened your pawn structure. The engine suggests {best_san} instead."
                                )
                            else:  # Pawn push
                                comment_bits.append(
                                    f"This pawn push overextended your position and weakened key squares. Consider {best_san} to maintain better structure."
                                )
                        else:
                            comment_bits.append(
                                f"Pawn moves require careful consideration of structure. {best_san} would have been more solid."
                            )

                    # King safety analysis
                    elif piece_type == chess.KING:
                        if delta_cp >= mis_t:
                            comment_bits.append(
                                f"Moving the king exposed it to danger. {best_san} would have been safer."
                            )
                        else:
                            comment_bits.append(
                                f"King safety is crucial. {best_san} would have been more prudent."
                            )

                    # Queen analysis
                    elif piece_type == chess.QUEEN:
                        if ply_idx <= 12 and delta_cp >= mis_t:
                            comment_bits.append(
                                f"Early queen moves often lose tempi and expose the queen to attack. {best_san} would develop a minor piece first."
                            )
                        elif delta_cp >= mis_t:
                            comment_bits.append(
                                f"The queen move was too aggressive and exposed it to threats. {best_san} would have been more positional."
                            )
                        else:
                            comment_bits.append(
                                f"Queen moves need careful timing. {best_san} would have been more accurate."
                            )

                    # Minor pieces analysis
                    elif piece_type in [chess.KNIGHT, chess.BISHOP]:
                        if delta_cp >= mis_t:
                            comment_bits.append(
                                f"This {chess.piece_name(piece_type).lower()} move was inaccurate and lost material/position. {best_san} would have been better."
                            )
                        else:
                            comment_bits.append(
                                f"The {chess.piece_name(piece_type).lower()} move wasn't optimal. {best_san} would have been more precise."
                            )

                    # Rook analysis
                    elif piece_type == chess.ROOK:
                        if delta_cp >= mis_t:
                            comment_bits.append(
                                f"The rook move was premature and didn't improve your position. {best_san} would have been more effective."
                            )
                        else:
                            comment_bits.append(
                                f"Rook moves should be well-timed. {best_san} would have been more accurate."
                            )

                # General positional advice based on delta
                if delta_cp >= 300:  # Blunder
                    comment_bits.append(
                        f"This was a serious blunder that lost significant advantage. The engine's line {best_line} shows the correct continuation."
                    )
                elif delta_cp >= 150:  # Mistake
                    comment_bits.append(
                        f"This mistake gave your opponent a clear advantage. Study the engine's suggestion: {best_line}"
                    )
                else:  # Inaccuracy
                    comment_bits.append(
                        f"This inaccuracy slightly worsened your position. {best_san} would have maintained equality."
                    )

                # Add general advice if no specific comments
                if not comment_bits:
                    comment_bits.append(
                        f"Consider the engine's suggested line: {best_line}"
                    )

                comment = " ".join(comment_bits)

                notes.append(
                    CoachNote(
                        move_no=move_no,
                        side=mover_name,
                        tag=tag,
                        delta_pawns=round(delta_cp / 100.0, 2),
                        played_san=played_san,
                        best_san=best_san,
                        best_line=best_line,
                        comment=comment,
                    )
                )
        return notes
    finally:
        engine.quit()


# ---------------------------
# Pattern Analysis Functions
# ---------------------------


def _analyze_tactical_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze tactical mistakes and piece losses."""
    patterns = []

    # Group by piece type and mistake severity
    piece_losses = defaultdict(lambda: defaultdict(list))
    for note in notes:
        if note.delta_pawns >= 1.5:  # Significant material loss
            move_san = note.played_san
            if move_san[0].isupper():  # Piece move
                piece_type = move_san[0]
                piece_losses[piece_type][note.tag].append(note)

    # Analyze each piece type
    for piece_type, mistakes_by_severity in piece_losses.items():
        total_mistakes = sum(
            len(mistakes) for mistakes in mistakes_by_severity.values()
        )
        if total_mistakes >= 3:  # Only report if significant pattern
            blunders = mistakes_by_severity.get("Blunder", [])
            mistakes = mistakes_by_severity.get("Mistake", [])

            # Analyze specific patterns for each piece
            if piece_type == "N":  # Knight
                patterns.append(
                    MistakePattern(
                        pattern_type=f"Knight losses",
                        frequency=total_mistakes,
                        total_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ),
                        avg_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        )
                        / total_mistakes,
                        examples=[
                            m.played_san
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ][:5],
                        description=f"You lose your knight {total_mistakes} times, often by moving it to squares where it can be trapped by pawns or pinned pieces.",
                    )
                )
            elif piece_type == "Q":  # Queen
                early_queen = [
                    m
                    for mistakes in mistakes_by_severity.values()
                    for m in mistakes
                    if m.move_no <= 15
                ]
                if len(early_queen) >= 2:
                    patterns.append(
                        MistakePattern(
                            pattern_type=f"Early Queen losses",
                            frequency=len(early_queen),
                            total_delta=sum(m.delta_pawns for m in early_queen),
                            avg_delta=sum(m.delta_pawns for m in early_queen)
                            / len(early_queen),
                            examples=[m.played_san for m in early_queen][:3],
                            description=f"You lose your queen {len(early_queen)} times in the opening by moving it too early, allowing opponent to develop with tempo.",
                        )
                    )
            elif piece_type == "B":  # Bishop
                patterns.append(
                    MistakePattern(
                        pattern_type=f"Bishop losses",
                        frequency=total_mistakes,
                        total_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ),
                        avg_delta=sum(
                            m.delta_pawns
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        )
                        / total_mistakes,
                        examples=[
                            m.played_san
                            for mistakes in mistakes_by_severity.values()
                            for m in mistakes
                        ][:5],
                        description=f"You lose your bishop {total_mistakes} times, often by pinning it to your king without an escape square or moving it to trapped squares.",
                    )
                )

    return patterns


def _analyze_positional_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze positional mistakes."""
    patterns = []

    # King safety issues
    king_safety_mistakes = [
        n
        for n in notes
        if "O-O" in n.played_san or "O-O-O" in n.played_san or "K" in n.played_san
    ]
    if len(king_safety_mistakes) >= 3:
        patterns.append(
            MistakePattern(
                pattern_type="King safety issues",
                frequency=len(king_safety_mistakes),
                total_delta=sum(m.delta_pawns for m in king_safety_mistakes),
                avg_delta=sum(m.delta_pawns for m in king_safety_mistakes)
                / len(king_safety_mistakes),
                examples=[m.played_san for m in king_safety_mistakes][:5],
                description=f"You make {len(king_safety_mistakes)} king safety mistakes, often by moving the f-pawn without castling first or exposing your king to attacks.",
            )
        )

    # Pawn structure issues
    pawn_mistakes = [
        n for n in notes if n.played_san[0].islower() and n.delta_pawns >= 0.5
    ]
    if len(pawn_mistakes) >= 5:
        patterns.append(
            MistakePattern(
                pattern_type="Pawn structure mistakes",
                frequency=len(pawn_mistakes),
                total_delta=sum(m.delta_pawns for m in pawn_mistakes),
                avg_delta=sum(m.delta_pawns for m in pawn_mistakes)
                / len(pawn_mistakes),
                examples=[m.played_san for m in pawn_mistakes][:5],
                description=f"You make {len(pawn_mistakes)} pawn structure mistakes, often by creating isolated pawns, doubled pawns, or weakening key squares.",
            )
        )

    return patterns


def _analyze_opening_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze opening-specific mistakes."""
    patterns = []

    opening_mistakes = [n for n in notes if n.move_no <= 15]
    if len(opening_mistakes) >= 5:
        # Early queen moves
        early_queen = [n for n in opening_mistakes if "Q" in n.played_san]
        if len(early_queen) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Early Queen moves",
                    frequency=len(early_queen),
                    total_delta=sum(m.delta_pawns for m in early_queen),
                    avg_delta=sum(m.delta_pawns for m in early_queen)
                    / len(early_queen),
                    examples=[m.played_san for m in early_queen][:3],
                    description=f"You move your queen too early {len(early_queen)} times in the opening, losing tempi and exposing it to attacks.",
                )
            )

        # Premature attacks
        aggressive_moves = [
            n
            for n in opening_mistakes
            if n.delta_pawns >= 1.0 and ("x" in n.played_san or "+" in n.played_san)
        ]
        if len(aggressive_moves) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Premature attacks",
                    frequency=len(aggressive_moves),
                    total_delta=sum(m.delta_pawns for m in aggressive_moves),
                    avg_delta=sum(m.delta_pawns for m in aggressive_moves)
                    / len(aggressive_moves),
                    examples=[m.played_san for m in aggressive_moves][:3],
                    description=f"You make {len(aggressive_moves)} premature attacks in the opening before completing development, weakening your position.",
                )
            )

    return patterns


def _analyze_endgame_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze endgame-specific mistakes."""
    patterns = []

    endgame_mistakes = [n for n in notes if n.move_no >= 30]
    if len(endgame_mistakes) >= 5:
        # King activity in endgame
        king_moves = [n for n in endgame_mistakes if "K" in n.played_san]
        if len(king_moves) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Endgame King activity",
                    frequency=len(king_moves),
                    total_delta=sum(m.delta_pawns for m in king_moves),
                    avg_delta=sum(m.delta_pawns for m in king_moves) / len(king_moves),
                    examples=[m.played_san for m in king_moves][:3],
                    description=f"You make {len(king_moves)} king activity mistakes in the endgame, often by not following the rule of the square or not centralizing your king.",
                )
            )

        # Pawn endgame mistakes
        pawn_endgame = [
            n
            for n in endgame_mistakes
            if n.played_san[0].islower() and n.delta_pawns >= 1.0
        ]
        if len(pawn_endgame) >= 3:
            patterns.append(
                MistakePattern(
                    pattern_type="Pawn endgame mistakes",
                    frequency=len(pawn_endgame),
                    total_delta=sum(m.delta_pawns for m in pawn_endgame),
                    avg_delta=sum(m.delta_pawns for m in pawn_endgame)
                    / len(pawn_endgame),
                    examples=[m.played_san for m in pawn_endgame][:3],
                    description=f"You make {len(pawn_endgame)} pawn endgame mistakes, often by not calculating pawn races correctly or not using opposition.",
                )
            )

    return patterns


def _analyze_calculation_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """Analyze calculation and tactical mistakes."""
    patterns = []

    # Missed tactics
    tactical_mistakes = [
        n
        for n in notes
        if n.delta_pawns >= 2.0 and ("x" in n.played_san or "+" in n.played_san)
    ]
    if len(tactical_mistakes) >= 3:
        patterns.append(
            MistakePattern(
                pattern_type="Missed tactics",
                frequency=len(tactical_mistakes),
                total_delta=sum(m.delta_pawns for m in tactical_mistakes),
                avg_delta=sum(m.delta_pawns for m in tactical_mistakes)
                / len(tactical_mistakes),
                examples=[m.played_san for m in tactical_mistakes][:3],
                description=f"You miss {len(tactical_mistakes)} simple tactics because you don't check for checks, captures, and threats systematically.",
            )
        )

    # Back-rank issues
    back_rank_mistakes = [
        n
        for n in notes
        if "R" in n.played_san and n.move_no >= 20 and n.delta_pawns >= 1.0
    ]
    if len(back_rank_mistakes) >= 2:
        patterns.append(
            MistakePattern(
                pattern_type="Back-rank weaknesses",
                frequency=len(back_rank_mistakes),
                total_delta=sum(m.delta_pawns for m in back_rank_mistakes),
                avg_delta=sum(m.delta_pawns for m in back_rank_mistakes)
                / len(back_rank_mistakes),
                examples=[m.played_san for m in back_rank_mistakes][:3],
                description=f"You create {len(back_rank_mistakes)} back-rank weaknesses by moving your rook without preparing an escape square for your king.",
            )
        )

    return patterns


def _identify_patterns(notes: List[CoachNote]) -> List[MistakePattern]:
    """
    Identify specific behavioral patterns from a list of coaching notes.
    """
    patterns = []

    # Group by mistake type and analyze specific behaviors
    by_type = defaultdict(list)
    for note in notes:
        by_type[note.tag].append(note)

    # Analyze specific behavioral patterns
    patterns.extend(_analyze_tactical_patterns(notes))
    patterns.extend(_analyze_positional_patterns(notes))
    patterns.extend(_analyze_opening_patterns(notes))
    patterns.extend(_analyze_endgame_patterns(notes))
    patterns.extend(_analyze_calculation_patterns(notes))

    # Sort by frequency and impact
    patterns.sort(key=lambda p: (p.frequency, p.avg_delta), reverse=True)
    return patterns[:15]  # Return top 15 patterns


# ---------------------------
# MCP Server Setup
# ---------------------------

mcp = FastMCP("Chess Analysis Server")


@mcp.tool(
    title="Monthly Chess Analysis",
    description="Analyze all games from the last 30 days and identify recurring mistake patterns to help improve your chess game"
)
def get_monthly_analysis(
    username: str, engine_path: str, depth: int = 12, max_games: int = 20
) -> List[Dict]:
    """
    Analyze all games from the last 30 days and return behavioral patterns.

    This function identifies recurring mistakes and weaknesses in your chess play
    over the past month, providing actionable insights for improvement.

    Args:
        username (str): Chess.com username to analyze
        engine_path (str): Path to Stockfish engine executable
        depth (int): Analysis depth for engine (higher = more accurate, slower)
        max_games (int): Maximum number of games to analyze (for performance)

    Returns:
        List[Dict]: List of identified mistake patterns with:
            - pattern_type: Type of mistake pattern
            - frequency: How many times it occurred
            - avg_delta: Average centipawn loss
            - examples: Specific moves that caused the mistakes
            - description: Actionable advice for improvement
    """
    print(f"Fetching games from last 30 days for {username}...")
    pgns = get_games_last_30_days(username)
    print(f"Found {len(pgns)} games, analyzing max {max_games}...")

    # Limit to max_games for performance
    pgns = pgns[:max_games]

    all_notes = []
    for i, pgn in enumerate(pgns):
        print(f"Analyzing game {i+1}/{len(pgns)}...")
        try:
            notes = analyze_pgn_with_stockfish(
                pgn, engine_path=engine_path, depth=depth
            )
            all_notes.extend(notes)
        except Exception as e:
            print(f"Error analyzing game {i+1}: {e}")
            continue

    print(f"Total mistakes found: {len(all_notes)}")
    patterns = _identify_patterns(all_notes)

    # Convert to dict format for JSON serialization
    return [
        {
            "pattern_type": p.pattern_type,
            "frequency": p.frequency,
            "total_delta": p.total_delta,
            "avg_delta": p.avg_delta,
            "examples": p.examples,
            "description": p.description,
        }
        for p in patterns
    ]


@mcp.tool(
    title="Human Chess Feedback",
    description="Analyze a single game and provide detailed human-readable feedback explaining mistakes and better alternatives"
)
def get_human_feedback(pgn_text: str, engine_path: str, depth: int = 16) -> List[Dict]:
    """
    Analyze a single game and return detailed human-readable feedback.

    This function provides educational comments explaining why moves were
    problematic and what better alternatives exist, helping you understand
    chess principles and improve your play.

    Args:
        pgn_text (str): PGN text of the game to analyze
        engine_path (str): Path to Stockfish engine executable
        depth (int): Analysis depth for engine (higher = more accurate, slower)

    Returns:
        List[Dict]: List of coaching notes with:
            - move_no: Move number in the game
            - side: "White" or "Black"
            - tag: "Blunder", "Mistake", or "Inaccuracy"
            - delta_pawns: Centipawn loss from the mistake
            - played_san: The move that was played
            - best_san: The best move according to engine
            - best_line: The engine's suggested continuation
            - comment: Detailed human-readable explanation
    """
    notes = analyze_pgn_with_stockfish(pgn_text, engine_path, depth)

    # Convert to dict format for JSON serialization
    return [
        {
            "move_no": note.move_no,
            "side": note.side,
            "tag": note.tag,
            "delta_pawns": note.delta_pawns,
            "played_san": note.played_san,
            "best_san": note.best_san,
            "best_line": note.best_line,
            "comment": note.comment,
        }
        for note in notes
    ]


@mcp.tool(
    title="Classic Chess Analysis",
    description="Analyze a single game and return classic chess.com style analysis with move classifications and centipawn values"
)
def get_classic_analysis(
    pgn_text: str, engine_path: str, depth: int = 16
) -> List[Dict]:
    """
    Analyze a single game and return classic chess.com style analysis.

    This function provides the traditional analysis format showing
    inaccuracies, mistakes, and blunders with their centipawn values,
    similar to chess.com's game analysis feature.

    Args:
        pgn_text (str): PGN text of the game to analyze
        engine_path (str): Path to Stockfish engine executable
        depth (int): Analysis depth for engine (higher = more accurate, slower)

    Returns:
        List[Dict]: List of analysis results with:
            - move: Move number and notation
            - evaluation: Centipawn evaluation after the move
            - accuracy: Move accuracy percentage
            - classification: "brilliant", "great", "good", "inaccuracy", "mistake", "blunder"
            - centipawn_loss: Centipawn value lost by the move
    """
    notes = analyze_pgn_with_stockfish(pgn_text, engine_path, depth)

    # Convert to classic analysis format
    classic_analysis = []
    for note in notes:
        # Determine classification based on centipawn loss
        cp_loss = int(note.delta_pawns * 100)
        if cp_loss >= 300:
            classification = "blunder"
        elif cp_loss >= 150:
            classification = "mistake"
        elif cp_loss >= 50:
            classification = "inaccuracy"
        elif cp_loss <= -100:
            classification = "brilliant"
        elif cp_loss <= -50:
            classification = "great"
        else:
            classification = "good"

        classic_analysis.append(
            {
                "move": f"{note.move_no}. {note.played_san}",
                "evaluation": f"{cp_loss:+d}cp",
                "accuracy": max(0, 100 - cp_loss),
                "classification": classification,
                "centipawn_loss": cp_loss,
                "best_move": note.best_san,
                "best_line": note.best_line,
            }
        )

    return classic_analysis


@mcp.tool(
    title="Latest Game Analysis",
    description="Fetch and analyze the most recent finished game for a user with detailed feedback"
)
def get_latest_game_analysis(username: str, engine_path: str, depth: int = 16) -> Dict:
    """
    Fetch and analyze the most recent finished game for a user.

    This is a convenience function that combines fetching the latest game
    with detailed human-readable analysis.

    Args:
        username (str): Chess.com username
        engine_path (str): Path to Stockfish engine executable
        depth (int): Analysis depth for engine

    Returns:
        Dict: Contains 'pgn' and 'notes' keys with the game data and analysis
    """
    pgn = get_latest_game_pgn(username)
    notes = get_human_feedback(pgn, engine_path, depth)

    return {
        "pgn": pgn,
        "notes": [
            {
                "move_no": note.move_no,
                "side": note.side,
                "tag": note.tag,
                "delta_pawns": note.delta_pawns,
                "played_san": note.played_san,
                "best_san": note.best_san,
                "best_line": note.best_line,
                "comment": note.comment,
            }
            for note in notes
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
