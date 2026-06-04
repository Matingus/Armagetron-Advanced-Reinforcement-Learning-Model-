"""
Armagetron Advanced Ladderlog Parser
=====================================
Parses events emitted by the dedicated server via CONSOLE_LADDER_LOG 1.
Each line arrives prefixed with '[L] ' and maps to a structured LadderlogEvent.

Reference: https://wiki.armagetronad.org/index.php?title=Styctap/Ladderlog_Events
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Event type registry
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Position / state
    PLAYER_GRIDPOS        = "PLAYER_GRIDPOS"
    ZONE_GRIDPOS          = "ZONE_GRIDPOS"
    ZONE_CREATED          = "ZONE_CREATED"

    # Player lifecycle
    PLAYER_ENTERED_GRID   = "PLAYER_ENTERED_GRID"
    PLAYER_ENTERED_SPECTATOR = "PLAYER_ENTERED_SPECTATOR"
    PLAYER_LEFT           = "PLAYER_LEFT"
    PLAYER_RENAMED        = "PLAYER_RENAMED"
    ONLINE_PLAYER         = "ONLINE_PLAYER"
    ONLINE_AI             = "ONLINE_AI"

    # Deaths / scoring
    DEATH_FRAG            = "DEATH_FRAG"
    DEATH_SUICIDE         = "DEATH_SUICIDE"
    DEATH_TEAMKILL        = "DEATH_TEAMKILL"
    DEATH_DEATHZONE       = "DEATH_DEATHZONE"
    ROUND_SCORE           = "ROUND_SCORE"
    ROUND_SCORE_TEAM      = "ROUND_SCORE_TEAM"

    # Round / match flow
    ROUND_STARTED         = "ROUND_STARTED"
    ROUND_ENDED           = "ROUND_ENDED"
    NEW_ROUND             = "NEW_ROUND"
    ROUND_COMMENCING      = "ROUND_COMMENCING"
    MATCH_ENDED           = "MATCH_ENDED"

    # Timing
    GAME_TIME             = "GAME_TIME"

    # Misc
    CURRENT_MAP           = "CURRENT_MAP"
    NUM_HUMANS            = "NUM_HUMANS"
    ALIVE                 = "ALIVE"
    WAIT_FOR_EXTERNAL_SCRIPT = "WAIT_FOR_EXTERNAL_SCRIPT"

    UNKNOWN               = "UNKNOWN"


# ---------------------------------------------------------------------------
# Event data container
# ---------------------------------------------------------------------------

@dataclass
class LadderlogEvent:
    type: EventType
    raw: str
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<LadderlogEvent {self.type.value} {self.data}>"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_event(line: str) -> Optional[LadderlogEvent]:
    """
    Parse one ladderlog line (already stripped of the '[L] ' prefix) into
    a LadderlogEvent.  Returns None for blank lines.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if not parts:
        return None

    tag = parts[0]

    # Resolve event type
    try:
        etype = EventType(tag)
    except ValueError:
        return LadderlogEvent(type=EventType.UNKNOWN, raw=line, data={"raw": line})

    data: dict = {}

    # ------------------------------------------------------------------
    # PLAYER_GRIDPOS  log_name  posx posy dirx diry speed [rubber_used rubber_total]
    # ------------------------------------------------------------------
    if etype == EventType.PLAYER_GRIDPOS:
        if len(parts) >= 7:
            data = {
                "name":         parts[1],
                "pos_x":        _f(parts[2]),
                "pos_y":        _f(parts[3]),
                "dir_x":        _f(parts[4]),
                "dir_y":        _f(parts[5]),
                "speed":        _f(parts[6]),
                "rubber_used":  _f(parts[7]) if len(parts) > 7 else 0.0,
                "rubber_total": _f(parts[8]) if len(parts) > 8 else 100.0,
            }

    # ------------------------------------------------------------------
    # PLAYER_ENTERED_GRID  log_name  ip_address  screen_name
    # ------------------------------------------------------------------
    elif etype == EventType.PLAYER_ENTERED_GRID:
        if len(parts) >= 2:
            data = {
                "log_name":    parts[1],
                "ip":          parts[2] if len(parts) > 2 else "",
                "screen_name": " ".join(parts[3:]) if len(parts) > 3 else parts[1],
            }

    # ------------------------------------------------------------------
    # PLAYER_ENTERED_SPECTATOR  log_name  ip  screen_name
    # ------------------------------------------------------------------
    elif etype == EventType.PLAYER_ENTERED_SPECTATOR:
        if len(parts) >= 2:
            data = {
                "log_name":    parts[1],
                "ip":          parts[2] if len(parts) > 2 else "",
                "screen_name": " ".join(parts[3:]) if len(parts) > 3 else parts[1],
            }

    # ------------------------------------------------------------------
    # PLAYER_LEFT  log_name
    # ------------------------------------------------------------------
    elif etype == EventType.PLAYER_LEFT:
        if len(parts) >= 2:
            data = {"name": parts[1]}

    # ------------------------------------------------------------------
    # PLAYER_RENAMED  old_name  new_name  ip  did_login  screen_name
    # ------------------------------------------------------------------
    elif etype == EventType.PLAYER_RENAMED:
        if len(parts) >= 3:
            data = {
                "old_name": parts[1],
                "new_name": parts[2],
                "ip": parts[3] if len(parts) > 3 else "",
            }

    # ------------------------------------------------------------------
    # ONLINE_PLAYER  name  ping  team  access_level  total_score  color  screen_name+
    # ONLINE_AI      name  team  total_score  color  screen_name+
    # Both are used to detect players present in the game.
    # ------------------------------------------------------------------
    elif etype in (EventType.ONLINE_PLAYER, EventType.ONLINE_AI):
        if len(parts) >= 2:
            data = {
                "name": parts[1],
                # ONLINE_PLAYER: team is parts[3]; ONLINE_AI: team is parts[2]
                "team": parts[3] if (etype == EventType.ONLINE_PLAYER and len(parts) > 3)
                        else (parts[2] if len(parts) > 2 else ""),
                "is_ai": etype == EventType.ONLINE_AI,
            }

    # ------------------------------------------------------------------
    # DEATH_FRAG  killed  killer
    # ------------------------------------------------------------------
    elif etype == EventType.DEATH_FRAG:
        if len(parts) >= 3:
            data = {"killed": parts[1], "killer": parts[2]}
        elif len(parts) >= 2:
            data = {"killed": parts[1], "killer": None}

    # ------------------------------------------------------------------
    # DEATH_SUICIDE  player
    # ------------------------------------------------------------------
    elif etype in (EventType.DEATH_SUICIDE, EventType.DEATH_DEATHZONE):
        if len(parts) >= 2:
            data = {"killed": parts[1]}

    # ------------------------------------------------------------------
    # DEATH_TEAMKILL  killed  killer
    # ------------------------------------------------------------------
    elif etype == EventType.DEATH_TEAMKILL:
        if len(parts) >= 3:
            data = {"killed": parts[1], "killer": parts[2]}

    # ------------------------------------------------------------------
    # ROUND_SCORE  player  score  team
    # ------------------------------------------------------------------
    elif etype == EventType.ROUND_SCORE:
        if len(parts) >= 3:
            data = {"player": parts[1], "score": _i(parts[2]),
                    "team": parts[3] if len(parts) > 3 else ""}

    # ------------------------------------------------------------------
    # ROUND_STARTED / ROUND_ENDED  time
    # ------------------------------------------------------------------
    elif etype in (EventType.ROUND_STARTED, EventType.ROUND_ENDED,
                   EventType.MATCH_ENDED):
        data = {"time": _f(parts[1]) if len(parts) > 1 else 0.0}

    # ------------------------------------------------------------------
    # GAME_TIME  elapsed
    # ------------------------------------------------------------------
    elif etype == EventType.GAME_TIME:
        data = {"time": _f(parts[1]) if len(parts) > 1 else 0.0}

    # ------------------------------------------------------------------
    # ALIVE  count
    # ------------------------------------------------------------------
    elif etype == EventType.ALIVE:
        data = {"count": _i(parts[1]) if len(parts) > 1 else 0}

    # ------------------------------------------------------------------
    # NUM_HUMANS  count
    # ------------------------------------------------------------------
    elif etype == EventType.NUM_HUMANS:
        data = {"count": _i(parts[1]) if len(parts) > 1 else 0}

    # ------------------------------------------------------------------
    # ZONE_GRIDPOS  effect id name radius expansion x y xdir ydir r g b
    # ------------------------------------------------------------------
    elif etype == EventType.ZONE_GRIDPOS:
        if len(parts) >= 8:
            data = {
                "effect": parts[1],
                "id":     parts[2],
                "name":   parts[3],
                "radius": _f(parts[4]),
                "x":      _f(parts[6]),
                "y":      _f(parts[7]),
            }

    # ------------------------------------------------------------------
    # ZONE_CREATED  effect id name x y xdir ydir
    # ------------------------------------------------------------------
    elif etype == EventType.ZONE_CREATED:
        if len(parts) >= 6:
            data = {
                "effect": parts[1],
                "id":     parts[2],
                "name":   parts[3],
                "x":      _f(parts[4]),
                "y":      _f(parts[5]),
            }

    # ------------------------------------------------------------------
    # CURRENT_MAP  size_factor  size_multiplier  map_file
    # ------------------------------------------------------------------
    elif etype == EventType.CURRENT_MAP:
        data = {"map_file": parts[3] if len(parts) > 3 else ""}

    return LadderlogEvent(type=etype, raw=line, data=data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _i(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def is_death_event(event: LadderlogEvent) -> bool:
    return event.type in (
        EventType.DEATH_FRAG,
        EventType.DEATH_SUICIDE,
        EventType.DEATH_TEAMKILL,
        EventType.DEATH_DEATHZONE,
    )


def is_round_end_event(event: LadderlogEvent) -> bool:
    return event.type in (
        EventType.ROUND_ENDED,
        EventType.MATCH_ENDED,
    )
