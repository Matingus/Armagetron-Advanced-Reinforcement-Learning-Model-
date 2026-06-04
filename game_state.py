"""
Armagetron Advanced – Live Game State
=======================================
Maintained by the background I/O thread.  All mutations happen there;
the Gymnasium step() thread only reads (Python GIL ensures safe access to
simple attribute reads/appends on CPython).

Wall tracking strategy
-----------------------
Every PLAYER_GRIDPOS emits the cycle's current (x, y).  Between two
consecutive positions we store a wall segment.  These segments, together
with the four arena boundary edges, form the obstacle set used for
ray-casting inside ObservationBuilder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Segment = (x0, y0, x1, y1)
Segment = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Per-player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    name: str
    pos_x: float = 0.0
    pos_y: float = 0.0
    dir_x: float = 1.0
    dir_y: float = 0.0
    speed: float = 0.0
    rubber_used: float = 0.0
    rubber_total: float = 100.0
    alive: bool = False
    is_ai: bool = False
    # Ordered list of (x, y) visited this round — used to build wall segments.
    _positions: List[Tuple[float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[float, float]:
        return (self.pos_x, self.pos_y)

    @property
    def direction(self) -> Tuple[float, float]:
        return (self.dir_x, self.dir_y)

    @property
    def heading_angle(self) -> float:
        """Heading in radians, 0 = +X, CCW positive."""
        return math.atan2(self.dir_y, self.dir_x)

    @property
    def rubber_fraction(self) -> float:
        """Remaining rubber as a fraction [0, 1]."""
        if self.rubber_total <= 0:
            return 0.0
        used = max(0.0, min(self.rubber_used, self.rubber_total))
        return 1.0 - used / self.rubber_total

    def new_round(self) -> None:
        """Clear per-round position history; keep identity."""
        self._positions.clear()
        self.alive = True


# ---------------------------------------------------------------------------
# Arena-wide game state
# ---------------------------------------------------------------------------

class GameState:
    """
    Thread-shared, mutable game state.

    Typical lifetime
    ----------------
    reset()              → called at the start of every Gymnasium episode
    update_*()           → called by the background reader thread as events arrive
    wall_segments        → read by ObservationBuilder inside step()
    """

    def __init__(
        self,
        arena_size: float = 200.0,
        max_trail_per_player: int = 4_000,
    ) -> None:
        self.arena_size = arena_size
        self.half = arena_size / 2.0
        self.max_trail = max_trail_per_player

        # Filled by PLAYER_ENTERED_GRID / PLAYER_GRIDPOS events.
        self.players: Dict[str, PlayerState] = {}

        self.game_time: float = 0.0
        self.round_active: bool = False
        self.alive_count: int = 0

        # Accumulated wall segments for ALL players this round.
        self._wall_segments: List[Segment] = []

        # Static arena boundary walls (counter-clockwise).
        h = self.half
        self._arena_walls: List[Segment] = [
            (-h, -h,  h, -h),   # south wall
            ( h, -h,  h,  h),   # east wall
            ( h,  h, -h,  h),   # north wall
            (-h,  h, -h, -h),   # west wall
        ]

    # ------------------------------------------------------------------
    # Full reset (called once per Gymnasium episode)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.players.clear()
        self._wall_segments.clear()
        self.game_time = 0.0
        self.round_active = False
        self.alive_count = 0

    # ------------------------------------------------------------------
    # Called by the background reader thread on ROUND_STARTED
    # ------------------------------------------------------------------

    def reset_round(self) -> None:
        """Keep players but wipe trail data for the new round."""
        self._wall_segments.clear()
        for p in self.players.values():
            p.new_round()
        self.round_active = True

    # ------------------------------------------------------------------
    # Event-driven state updates
    # ------------------------------------------------------------------

    def update_player_position(
        self,
        name: str,
        pos_x: float,
        pos_y: float,
        dir_x: float,
        dir_y: float,
        speed: float,
        rubber_used: float = 0.0,
        rubber_total: float = 100.0,
    ) -> None:
        """
        Called on every PLAYER_GRIDPOS event.
        Extends the wall with a new segment from the previous position.
        """
        if name not in self.players:
            self.players[name] = PlayerState(name=name)

        p = self.players[name]

        # Build wall segment from the previous known position.
        if p._positions:
            prev = p._positions[-1]
            # Only add a segment if the cycle actually moved.
            dx = pos_x - prev[0]
            dy = pos_y - prev[1]
            if dx * dx + dy * dy > 1e-6:
                self._wall_segments.append((prev[0], prev[1], pos_x, pos_y))

        # Update player record.
        p.pos_x = pos_x
        p.pos_y = pos_y
        p.dir_x = dir_x
        p.dir_y = dir_y
        p.speed = speed
        p.rubber_used = rubber_used
        p.rubber_total = rubber_total
        p.alive = True
        p._positions.append((pos_x, pos_y))

        # Bound trail memory.
        if len(p._positions) > self.max_trail:
            # Drop the oldest half.
            p._positions = p._positions[self.max_trail // 2 :]

        # Bound global wall segment list.
        if len(self._wall_segments) > self.max_trail * len(self.players) * 2:
            self._wall_segments = self._wall_segments[-(self.max_trail * len(self.players)):]

    def add_player(self, log_name: str, is_ai: bool = False) -> None:
        if log_name not in self.players:
            p = PlayerState(name=log_name, is_ai=is_ai)
            self.players[log_name] = p

    def remove_player(self, name: str) -> None:
        self.players.pop(name, None)

    def mark_player_dead(self, name: str) -> None:
        if name in self.players:
            self.players[name].alive = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def wall_segments(self) -> List[Segment]:
        """All current obstacles: player trails + arena boundary."""
        return self._wall_segments + self._arena_walls

    @property
    def alive_players(self) -> List[PlayerState]:
        return [p for p in self.players.values() if p.alive]

    def get_player(self, name: str) -> Optional[PlayerState]:
        return self.players.get(name)

    def player_alive(self, name: str) -> bool:
        p = self.players.get(name)
        return p.alive if p else False

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        alive = sum(1 for p in self.players.values() if p.alive)
        return (
            f"<GameState t={self.game_time:.1f}s "
            f"players={len(self.players)} alive={alive} "
            f"walls={len(self._wall_segments)}>"
        )
