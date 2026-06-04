"""
ArmagetronEnv – Gymnasium Environment for Armagetron Advanced
==============================================================

Communication model
-------------------
                  ┌─────────────────────────────────────┐
                  │  armagetronad-dedicated (subprocess) │
                  │                                      │
   stdin ◄────────│─── CYCLE_TURN / NEXT_ROUND / ADD_BOT │
                  │                                      │
   stdout ─────── │─── [L] PLAYER_GRIDPOS ...            │  CONSOLE_LADDER_LOG 1
                  │    [L] DEATH_FRAG ...                │
                  └─────────────────────────────────────┘
                        │
               background reader thread
                        │  ① calls _apply_event()  → updates GameState immediately
                        │  ② puts LadderlogEvent   → into event_queue
                        ▼
                   event_queue  ◄──── step() / _wait_for_round() drain

Agent control
-------------
  Action space  : Discrete(3)   0=turn-left  1=straight  2=turn-right
  Observation   : Box(n,)       see ObservationBuilder for layout

Bug fixes applied (vs original armagetron_env.py)
--------------------------------------------------
  FIX 1 – _reader_loop called `game_state.update_from_event()` which does not
           exist on GameState.  Replaced with `self._apply_event(event)`.
           Without this fix, game state was NEVER updated.

  FIX 2 – reset() called _wait_for_round() BEFORE _spawn_bots().
           No bots in the server → no round ever starts → 10 s timeout →
           server restart → same problem → RuntimeError.
           Fixed: spawn bots first (guarded by _bots_spawned flag), then wait.

  FIX 3 – _wait_for_round() re-queued every non-round event then immediately
           re-retrieved it, looping on the same event forever and starving the
           queue of real progress.
           Fixed: simply discard non-round events (game state is already updated
           by the reader thread before events enter the queue).

  FIX 4 – step() was missing entirely.  Added with proper obs/reward/done logic.

Additional improvements
-----------------------
  • threading.Event for server-ready detection (no bare sleep guessing)
  • _cmd() now writes bytes (subprocess stdin is binary on Python 3)
  • _detect_agent() reads directly from GameState instead of queue-walking
  • _wait_for_gridpos() polls GameState instead of queue-peeking
  • Cleaner reset() flow with recursion guard
  • metadata dict for Gymnasium compliance
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .ladderlog_parser import (
    EventType,
    LadderlogEvent,
    is_death_event,
    is_round_end_event,
    parse_event,
)
from .game_state import GameState
from .observation_builder import ObservationBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward shaping constants  (tune freely)
# ---------------------------------------------------------------------------

R_STEP_ALIVE    =  0.01   # small survival bonus every decision step
R_AGENT_DIES    = -1.0    # terminal penalty when agent is killed
R_AGENT_FRAGS   =  1.0    # reward per enemy the agent kills
R_ENEMY_DIES    =  0.2    # reward when any enemy dies (not by agent)
R_ROUND_WIN     =  3.0    # agent is the last cycle alive
R_ROUND_SURVIVE =  1.0    # round ends for other reason while agent is alive


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ArmagetronEnv(gym.Env):
    """Gymnasium environment wrapping an Armagetron Advanced dedicated server."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_executable: str = "armagetronad-dedicated",
        config_dir: Optional[str] = None,
        agent_name: Optional[str] = None,
        num_rays: int = 16,
        max_opponents: int = 3,
        arena_size: float = 200.0,
        step_timeout: float = 5.0,
        num_enemy_bots: int = 3,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.server_executable = server_executable
        self.config_dir = Path(config_dir or Path(__file__).parent.parent / "config")
        self.configured_agent_name = agent_name
        self.step_timeout = step_timeout
        self.num_enemy_bots = num_enemy_bots
        self.render_mode = render_mode

        # Game logic helpers
        self.game_state = GameState(arena_size=arena_size)
        self.obs_builder = ObservationBuilder(
            arena_size=arena_size,
            num_rays=num_rays,
            max_opponents=max_opponents,
        )

        # Gymnasium spaces
        obs_size = self.obs_builder.get_obs_size()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Server process state
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._event_queue: queue.Queue[LadderlogEvent] = queue.Queue()

        # FIX improvement: use a threading.Event to know when the server
        # is alive rather than sleeping for an arbitrary fixed duration.
        self._server_ready_event = threading.Event()

        # Episode tracking
        self.agent_name: Optional[str] = self.configured_agent_name
        self._agent_alive = False
        self._step_count = 0
        self._ep_reward = 0.0

        # Guards so we only ADD_BOT once per server lifetime
        self._bots_spawned = False

        # One-time server start
        self._start_server()

    # =======================================================================
    # Gymnasium API
    # =======================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._flush_queue()
        self._step_count = 0
        self._ep_reward = 0.0

        # ------------------------------------------------------------------
        # FIX 2: spawn bots BEFORE waiting for a round to start.
        #
        # Original code waited for ROUND_STARTED first, then spawned bots.
        # But without any players in the server a round never starts, so the
        # wait always timed out and crashed.
        # ------------------------------------------------------------------
        if not self._bots_spawned:
            logger.info("Waiting for server to emit its first output...")
            if not self._server_ready_event.wait(timeout=30.0):
                raise RuntimeError(
                    "Armagetron server produced no output within 30 s. "
                    "Check server_executable path and the config directory."
                )
            # Give the server a moment to finish printing its startup banner
            # before we start issuing commands.
            time.sleep(1.5)
            logger.info("Spawning %d bot(s)…", 1 + self.num_enemy_bots)
            self._spawn_bots()
            self._bots_spawned = True
        else:
            # Episode just ended — ask server to move on to the next round.
            # NEXT_ROUND is a valid console command since Armagetron 0.2.8.
            # If it isn't recognised by your build, the server will simply
            # wait for the round to end naturally (ROUND_PAUSE_TIME controls
            # the gap; set it to 2 in server_info.cfg for fast resets).
            self._cmd("NEXT_ROUND")

        # Wait for the new round to begin
        if not self._wait_for_round(timeout=40.0):
            logger.warning("Round did not start within timeout. Restarting server.")
            self._stop_server()
            self._start_server()
            self._bots_spawned = False
            # One recursive retry; if this also fails something is fundamentally
            # wrong with the server config.
            return self.reset(seed=seed, options=options)

        # Resolve the agent's log-name (only needed once per server lifetime)
        if self.agent_name is None:
            self.agent_name = self._detect_agent(timeout=10.0)
            if self.agent_name is None:
                logger.warning("Could not identify agent bot. Retrying reset.")
                time.sleep(1.0)
                return self.reset(seed=seed, options=options)

        self._agent_alive = True

        obs = self._wait_for_gridpos(timeout=10.0)
        info = {
            "agent_name": self.agent_name,
            "step": 0,
            "episode_reward": 0.0,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        FIX 4: step() was completely missing from the original file.

        Executes one decision step:
          1. Send the chosen action to the server.
          2. Block until the agent's next PLAYER_GRIDPOS (or a terminal event).
          3. Build observation, compute reward, determine done flags.
          4. Return (obs, reward, terminated, truncated, info).
        """
        assert self.agent_name is not None, (
            "reset() must be called before step()."
        )

        self._step_count += 1

        # 1. Act
        self._execute_action(action)

        # 2. Wait for next state
        events, timed_out = self._drain_until_gridpos()

        # 3. Build observation from current game state
        obs = self.obs_builder.build_observation(self.game_state, self.agent_name)

        # 4. Reward & termination
        reward = self._compute_reward(events)
        terminated = self._compute_terminated(events, timed_out)
        truncated = False   # we don't enforce a wall-clock step limit here

        self._ep_reward += reward

        info = {
            "agent_name": self.agent_name,
            "step": self._step_count,
            "episode_reward": self._ep_reward,
            "alive": self._agent_alive,
            "timed_out": timed_out,
        }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Cleanly shut down the dedicated server subprocess."""
        self._stop_server()

    # =======================================================================
    # Action execution
    # =======================================================================

    def _execute_action(self, action: int) -> None:
        """
        Map Discrete(3) → CYCLE_TURN server command.

            0  → CYCLE_TURN <name>  1    (left)
            1  → (no command)            (straight)
            2  → CYCLE_TURN <name> -1    (right)
        """
        if self.agent_name is None:
            return
        if action == 0:
            self._cmd(f"CYCLE_TURN {self.agent_name} 1")
        elif action == 2:
            self._cmd(f"CYCLE_TURN {self.agent_name} -1")
        # action == 1: go straight, no command needed

    # =======================================================================
    # Event processing
    # =======================================================================

    def _drain_until_gridpos(
        self,
    ) -> Tuple[List[LadderlogEvent], bool]:
        """
        Pop events from the queue until:
          • We receive a PLAYER_GRIDPOS for our agent    → decision step done
          • A death event kills our agent                → episode terminates
          • A round-end event arrives                    → episode terminates
          • step_timeout elapses                         → timed_out=True

        Returns (collected_events, timed_out).
        """
        events: List[LadderlogEvent] = []
        deadline = time.monotonic() + self.step_timeout
        timed_out = False

        while time.monotonic() < deadline:
            try:
                ev = self._event_queue.get(timeout=0.02)
            except queue.Empty:
                continue

            events.append(ev)

            # Agent got a fresh position update → step complete
            if (
                ev.type == EventType.PLAYER_GRIDPOS
                and ev.data.get("name") == self.agent_name
            ):
                break

            # Agent died → episode ends
            if is_death_event(ev) and ev.data.get("killed") == self.agent_name:
                self._agent_alive = False
                break

            # Round / match finished
            if is_round_end_event(ev):
                break

        else:
            logger.debug("step() timed out waiting for PLAYER_GRIDPOS")
            timed_out = True

        return events, timed_out

    def _compute_reward(self, events: List[LadderlogEvent]) -> float:
        reward = R_STEP_ALIVE if self._agent_alive else 0.0

        for ev in events:
            if not is_death_event(ev):
                continue
            killed = ev.data.get("killed", "")
            killer = ev.data.get("killer")

            if killed == self.agent_name:
                reward += R_AGENT_DIES
            elif killer == self.agent_name:
                reward += R_AGENT_FRAGS
            elif killed and killed != self.agent_name:
                reward += R_ENEMY_DIES

        # Round ended while agent is alive
        for ev in events:
            if is_round_end_event(ev) and self._agent_alive:
                alive_count = len(self.game_state.alive_players)
                reward += R_ROUND_WIN if alive_count <= 1 else R_ROUND_SURVIVE

        return float(reward)

    def _compute_terminated(
        self, events: List[LadderlogEvent], timed_out: bool
    ) -> bool:
        if timed_out or not self._agent_alive:
            return True
        return any(is_round_end_event(ev) for ev in events)

    # =======================================================================
    # Server lifecycle
    # =======================================================================

    def _start_server(self) -> None:
        """Spawn armagetronad-dedicated with stdin/stdout pipes."""
        import time  # Ensure time is imported at the top of the file or here
        
        var_dir = self.config_dir.parent / "var"
        var_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.server_executable,
            "--userdatadir", str(self.config_dir),
            "--vardir",      str(var_dir),
        ]
        logger.info("Starting server: %s", " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr so we see everything
                bufsize=0,                 # unbuffered
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Server binary not found: '{self.server_executable}'.\n"
                "Install Armagetron Advanced Dedicated and update "
                "SERVER_EXE in train.py."
            ) from exc

        self._running = True
        self._server_ready_event.clear()

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="arma-reader",
            daemon=True,
        )
        self._reader_thread.start()

        # ===================================================================
        # ADDED: Force configurations via stdin right after the process starts
        # ===================================================================
        logger.info("Waiting briefly for server initialization...")
        time.sleep(0.5)  # Give the server 500ms to ready its console input pipe

        logger.info("Injecting fallback console configurations via stdin...")
        self._cmd("CONSOLE_LADDER_LOG 1")
        self._cmd("PLAYER_GRIDPOS_INTERVAL 0.05")
        self._cmd("SINGLE_PLAYER 1")
        self._cmd("START_ALONE 1")
        self._cmd("MIN_PLAYERS 4")
        self._cmd("AA_MIN_PLAYERS 4")
        # ===================================================================

    def _stop_server(self) -> None:
        if not self._proc:
            return

        self._running = False
        try:
            self._cmd("QUIT")
        except Exception:
            pass

        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        except Exception:
            pass

        self._proc = None

        if self._reader_thread:
            self._reader_thread.join(timeout=3)
            self._reader_thread = None

    # =======================================================================
    # Background reader thread — FIX 1 applied here
    # =======================================================================

    def _reader_loop(self) -> None:
        """
        Background thread: read server stdout line by line.

        FIX 1: original code called `self.game_state.update_from_event(event)`
        which does not exist on GameState and raised AttributeError silently
        (inside a daemon thread), leaving game state forever empty.

        Corrected order:
          ① parse the ladderlog line
          ② call _apply_event() to update GameState immediately
          ③ put the event into the queue for step() / _wait_for_round()
        """
        while self._running and self._proc and self._proc.stdout:
            raw_line = self._proc.stdout.readline()
            if not raw_line:
                # EOF — server process exited
                break

            # Signal the main thread that the server is alive
            self._server_ready_event.set()

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Only ladder-log lines are prefixed with "[L]"
            if line.startswith("[L]"):
                payload = line[3:].lstrip()
                event = parse_event(payload)
                if event:
                    # ① Update game state synchronously in this thread
                    self._apply_event(event)   # FIX 1
                    # ② Enqueue for step() / _wait_for_round()
                    self._event_queue.put(event)
            else:
                # ===================================================================
                # ADDED: Print any raw server logs that do not match the telemetry structure.
                # Changing level to INFO so you can see them on screen during troubleshooting.
                # ===================================================================
                logger.info("Server Raw Output: %s", line)

    # =======================================================================
    # Live state application   (called by reader thread)
    # =======================================================================

    def _apply_event(self, ev: LadderlogEvent) -> None:
        """
        Mutate GameState in response to a ladderlog event.
        Called from the reader thread, so keep this fast and lock-free.
        (Python's GIL makes simple attribute writes safe for the main thread
        to read concurrently.)
        """
        d = ev.data

        if ev.type == EventType.PLAYER_GRIDPOS:
            self.game_state.update_player_position(
                d["name"], d["pos_x"], d["pos_y"],
                d["dir_x"], d["dir_y"], d["speed"],
                d.get("rubber_used", 0.0),
                d.get("rubber_total", 100.0),
            )

        elif ev.type == EventType.PLAYER_ENTERED_GRID:
            name = d.get("log_name") or d.get("name", "")
            if name:
                self.game_state.add_player(name, is_ai=False)
                logger.debug("Player entered grid: %s", name)

        # Vanilla 0.2.9 uses ONLINE_PLAYER / ONLINE_AI instead of
        # PLAYER_ENTERED_GRID to announce players present in the game.
        elif ev.type in (EventType.ONLINE_PLAYER, EventType.ONLINE_AI):
            name = d.get("name", "")
            is_ai = d.get("is_ai", ev.type == EventType.ONLINE_AI)
            if name:
                self.game_state.add_player(name, is_ai=is_ai)
                logger.debug("Online player detected: %s (ai=%s)", name, is_ai)

        elif ev.type == EventType.PLAYER_LEFT:
            name = d.get("name", "")
            if name:
                self.game_state.remove_player(name)

        elif ev.type == EventType.PLAYER_RENAMED:
            old = d.get("old_name", "")
            new = d.get("new_name", "")
            if old in self.game_state.players and new:
                self.game_state.players[new] = self.game_state.players.pop(old)
                self.game_state.players[new].name = new
                # Keep our agent tracking up to date
                if self.agent_name == old:
                    self.agent_name = new
                    logger.info("Agent renamed: %s → %s", old, new)

        elif is_death_event(ev):
            killed = d.get("killed", "")
            if killed:
                self.game_state.mark_player_dead(killed)

        elif ev.type == EventType.GAME_TIME:
            self.game_state.game_time = d.get("time", 0.0)

        elif ev.type in (
            EventType.ROUND_STARTED,
            EventType.NEW_ROUND,
            EventType.ROUND_COMMENCING,
        ):
            self.game_state.reset_round()
            # NOTE: do NOT set self._agent_alive here — that belongs to
            # the main thread (reset() sets it after the round wait returns).

        elif ev.type in (EventType.ROUND_ENDED, EventType.MATCH_ENDED):
            self.game_state.round_active = False

    # =======================================================================
    # Bot management
    # =======================================================================

    def _spawn_bots(self) -> None:
        """
        Issue ADD_BOT commands to populate the server.
        Total bots = 1 (agent we control) + num_enemy_bots.

        The server assigns names like AI_1, AI_2 … in order.
        We treat AI_1 as the agent; all others are enemies.
        To use a different slot, pass agent_name='AI_2' etc.
        """
        total = 1 + self.num_enemy_bots
        for _ in range(total):
            self._cmd("ADD_BOT")
            time.sleep(0.15)
        logger.info("Spawned %d bot(s) total.", total)

    def _detect_agent(self, timeout: float = 10.0) -> Optional[str]:
        """
        Return the log-name of the bot we control.

        If configured_agent_name was given, return it directly.
        Otherwise wait up to `timeout` seconds for any player to appear
        in GameState (populated by the reader thread via _apply_event),
        then return the first one.
        """
        if self.configured_agent_name:
            return self.configured_agent_name

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # GameState.players is updated by the reader thread
            if self.game_state.players:
                name = next(iter(self.game_state.players))
                logger.info("Detected agent bot: '%s'", name)
                return name
            time.sleep(0.1)

        logger.warning("No players appeared in GameState after %.1fs.", timeout)
        return None

    # =======================================================================
    # Synchronisation helpers — FIX 3 applied to _wait_for_round
    # =======================================================================

    def _wait_for_round(self, timeout: float = 40.0) -> bool:
        """
        Block until a round-start event arrives in the event queue.

        FIX 3: original code put non-round events back into the queue with
        `self._event_queue.put(ev)`, then immediately re-retrieved them on the
        next iteration — creating an infinite busy-loop on the same event.

        Corrected approach: since the reader thread already called
        _apply_event() before enqueueing, GameState is up to date.
        Non-round events can simply be discarded here.
        """
        round_start_types = {
            EventType.ROUND_STARTED,
            EventType.NEW_ROUND,
            EventType.ROUND_COMMENCING,
        }
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                ev = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if ev.type in round_start_types:
                logger.info("Round started (%s).", ev.type.value)
                return True

            # FIX 3: discard — do NOT re-queue.
            # GameState was already updated when the event was parsed.

        logger.warning("_wait_for_round timed out after %.1f s.", timeout)
        return False

    def _wait_for_gridpos(self, timeout: float = 10.0) -> np.ndarray:
        """
        Poll GameState until the agent has at least one position entry,
        then return the initial observation.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.agent_name and self.game_state.get_player(self.agent_name):
                return self.obs_builder.build_observation(
                    self.game_state, self.agent_name
                )
            time.sleep(0.05)

        logger.warning(
            "No initial PLAYER_GRIDPOS for agent '%s' within %.1f s.",
            self.agent_name, timeout,
        )
        return np.zeros(self.obs_builder.get_obs_size(), dtype=np.float32)

    # =======================================================================
    # Utilities
    # =======================================================================

    def _cmd(self, command: str) -> None:
        """Write a console command to the server's stdin (as UTF-8 bytes)."""
        if self._proc and self._proc.stdin and not self._proc.stdin.closed:
            try:
                # subprocess.PIPE is a binary stream — must encode to bytes
                self._proc.stdin.write((command + "\n").encode("utf-8"))
                self._proc.stdin.flush()
                logger.debug("→ server: %s", command)
            except (BrokenPipeError, OSError) as exc:
                logger.warning("Failed to send command '%s': %s", command, exc)

    def _flush_queue(self) -> None:
        """Discard all pending events (called at the start of each episode)."""
        drained = 0
        while True:
            try:
                self._event_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.debug("Flushed %d stale event(s) from queue.", drained)

    # =======================================================================
    # Context manager support
    # =======================================================================

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"ArmagetronEnv(agent={self.agent_name!r}, "
            f"obs_size={self.obs_builder.get_obs_size()}, "
            f"step={self._step_count})"
        )
