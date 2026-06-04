"""
Observation Builder
===================
Converts live GameState into a flat numpy array for the policy network.

Observation layout (configurable, defaults shown)
--------------------------------------------------
  [0  : num_rays]           Ray distances in 16 directions (agent-relative, normalised)
  [num_rays : num_rays+6]   Agent features: pos_x, pos_y, dir_x, dir_y, speed, rubber_pct
  [... : ...+max_opp*5]     Up to max_opponents opponents × (pos_x, pos_y, dir_x, dir_y, alive)
  [-1]                      Normalised game time

All values are in [-1, 1] or [0, 1].

Ray-casting
-----------
For each of ``num_rays`` evenly-spaced angles (starting from the agent's
current heading and sweeping 360°), we test every wall segment and arena
boundary for intersection and record the normalised distance to the nearest
hit.  The algorithm is a standard 2-D parametric ray/segment intersection.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .game_state import GameState, PlayerState

# A wall segment is (x0, y0, x1, y1).
Segment = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Core 2-D ray ↔ segment intersection
# ---------------------------------------------------------------------------

def _ray_segment_t(
    ox: float, oy: float,   # ray origin
    dx: float, dy: float,   # ray direction (unit)
    ax: float, ay: float,   # segment start
    bx: float, by: float,   # segment end
) -> Optional[float]:
    """
    Return the parameter *t* ≥ 0 where the ray hits the segment, or None.

    Ray   : P(t) = (ox + t*dx,  oy + t*dy)       t ≥ 0
    Segment: Q(s) = (ax + s*(bx-ax), ay + s*(by-ay))  0 ≤ s ≤ 1

    Cross-product form avoids division by near-zero denominators gracefully.
    """
    # Segment direction
    ex = bx - ax
    ey = by - ay

    denom = dx * ey - dy * ex           # cross(ray_dir, seg_dir)
    if abs(denom) < 1e-12:
        return None                     # parallel (or degenerate segment)

    # Vector from ray origin to segment start
    fx = ax - ox
    fy = ay - oy

    t = (fx * ey - fy * ex) / denom
    s = (fx * dy - fy * dx) / denom

    if t >= 1e-6 and 0.0 <= s <= 1.0:
        return t
    return None


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """
    Builds a fixed-size numpy observation vector from a GameState snapshot.

    Parameters
    ----------
    arena_size   : Side length of the square arena (default 200 → ±100 each axis).
    num_rays     : Number of ray directions, evenly distributed over 360°.
    max_opponents: How many opponent slots to include in the observation.
    max_game_time: Expected maximum round duration in seconds (used for normalisation).
    """

    def __init__(
        self,
        arena_size: float = 200.0,
        num_rays: int = 16,
        max_opponents: int = 3,
        max_game_time: float = 120.0,
    ) -> None:
        self.half = arena_size / 2.0
        self.num_rays = num_rays
        self.max_opponents = max_opponents
        self.max_game_time = max_game_time

        # Longest possible ray distance across the diagonal of the arena.
        self.max_ray = math.sqrt(2.0) * arena_size

        # Ray angle offsets relative to the agent's heading (radians).
        self._angle_offsets = np.linspace(0.0, 2.0 * math.pi, num_rays, endpoint=False)

        # Observation vector length.
        self._obs_size = (
            num_rays                  # ray distances
            + 6                       # agent: pos_x, pos_y, dir_x, dir_y, speed, rubber
            + max_opponents * 5       # opponents: pos_x, pos_y, dir_x, dir_y, alive
            + 1                       # normalised game time
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_obs_size(self) -> int:
        return self._obs_size

    def build_observation(
        self,
        game_state: GameState,
        agent_name: str,
    ) -> np.ndarray:
        """
        Build and return the observation vector.
        Returns all-zeros if the agent is not in the game state.
        """
        agent = game_state.get_player(agent_name)
        if agent is None:
            return np.zeros(self._obs_size, dtype=np.float32)

        # --- 1. Ray distances -------------------------------------------
        rays = self._cast_rays(agent, game_state.wall_segments)

        # --- 2. Agent features ------------------------------------------
        agent_feat = np.array([
            np.clip(agent.pos_x / self.half, -1.0, 1.0),
            np.clip(agent.pos_y / self.half, -1.0, 1.0),
            agent.dir_x,                            # already ≈ unit vector
            agent.dir_y,
            np.clip(agent.speed / 50.0, 0.0, 1.0),  # rough max speed ~50
            agent.rubber_fraction,
        ], dtype=np.float32)

        # --- 3. Opponent features (closest first) -----------------------
        opp_feat = self._opponent_features(agent, game_state)

        # --- 4. Game time -----------------------------------------------
        t_norm = np.array(
            [np.clip(game_state.game_time / self.max_game_time, 0.0, 1.0)],
            dtype=np.float32,
        )

        return np.concatenate([rays, agent_feat, opp_feat, t_norm]).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cast_rays(
        self,
        agent: PlayerState,
        segments: List[Segment],
    ) -> np.ndarray:
        """
        Cast ``num_rays`` rays from the agent position and return normalised
        distances to the nearest obstacle for each ray.
        """
        ox, oy = agent.pos_x, agent.pos_y
        base_angle = agent.heading_angle

        distances = np.ones(self.num_rays, dtype=np.float32)

        for i, offset in enumerate(self._angle_offsets):
            angle = base_angle + offset
            dx = math.cos(angle)
            dy = math.sin(angle)

            nearest = self.max_ray

            for seg in segments:
                t = _ray_segment_t(ox, oy, dx, dy, seg[0], seg[1], seg[2], seg[3])
                if t is not None and t < nearest:
                    nearest = t

            distances[i] = nearest / self.max_ray  # normalise → [0, 1]

        return distances

    def _opponent_features(
        self,
        agent: PlayerState,
        game_state: GameState,
    ) -> np.ndarray:
        """
        Return a flattened array of per-opponent features.
        Opponents beyond ``max_opponents`` are dropped.
        Missing slots are zero-padded.
        """
        out = np.zeros(self.max_opponents * 5, dtype=np.float32)

        opponents = [
            p for name, p in game_state.players.items()
            if name != agent.name
        ]

        # Sort by Euclidean distance (closest first)
        opponents.sort(
            key=lambda p: (p.pos_x - agent.pos_x) ** 2
                        + (p.pos_y - agent.pos_y) ** 2
        )

        for slot, opp in enumerate(opponents[: self.max_opponents]):
            base = slot * 5
            out[base + 0] = np.clip(opp.pos_x / self.half, -1.0, 1.0)
            out[base + 1] = np.clip(opp.pos_y / self.half, -1.0, 1.0)
            out[base + 2] = opp.dir_x
            out[base + 3] = opp.dir_y
            out[base + 4] = 1.0 if opp.alive else 0.0

        return out
