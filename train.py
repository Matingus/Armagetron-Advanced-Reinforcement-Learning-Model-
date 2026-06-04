"""
train.py — PPO Training Entry Point for Armagetron Advanced RL Agent
=====================================================================

Quick-start checklist
---------------------
1.  Install Armagetron Advanced **Dedicated** (not the GUI client):
      https://launchpad.net/armagetronad/+download  (Windows installer)
      or build from source: https://gitlab.com/armagetronad/armagetronad

2.  Update SERVER_EXE below to point at your armagetronad_dedicated.exe.

3.  Copy  config/server_info.cfg  into your project's  config/  folder.
    (The path passed via --userdatadir must contain that file.)

4.  Run:
      python train.py

5.  Monitor training in TensorBoard:
      tensorboard --logdir tensorboard_logs

Agent name note
---------------
Armagetron assigns bots names like "AI_1", "AI_2" … in the order ADD_BOT
is issued.  The environment treats AI_1 as the controllable agent by
default (first player detected).  If you see "Agent not detected" warnings,
check the ladderlog output and pass the correct name:

    env = ArmagetronEnv(..., agent_name="AI_1")
"""

import logging
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.armagetron_env import ArmagetronEnv

# ---------------------------------------------------------------------------
# Logging: INFO shows round transitions; DEBUG shows every server event
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def make_env():
    """Factory function required by DummyVecEnv."""
    return ArmagetronEnv(
        server_executable=SERVER_EXE,
        config_dir=CONFIG_DIR,
        num_enemy_bots=3,        # 3 enemies + 1 agent = 4 bots total
        step_timeout=5.0,        # seconds to wait for each PLAYER_GRIDPOS
        num_rays=16,             # ray-cast directions in observation
        max_opponents=3,         # enemy slots in observation vector
        agent_name="teo",     # uncomment if auto-detection is unreliable
    )


def main():
    # -----------------------------------------------------------------------
    # 1.  Configuration — edit these two paths
    # -----------------------------------------------------------------------
    global SERVER_EXE, CONFIG_DIR

    SERVER_EXE = r"C:\Program Files (x86)\Armagetron Advanced Dedicated\armagetronad_dedicated.exe"
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "armagetron_data")

    # -----------------------------------------------------------------------
    # 2.  Environment
    # -----------------------------------------------------------------------
    logger.info("Initialising ArmagetronEnv…")
    env = make_env()

    # Validate the env against the Gymnasium API.
    # check_env calls reset() twice and step() several times —
    # the fixed env handles this correctly now.
    logger.info("Running SB3 environment checker…")
    check_env(env, warn=True)
    logger.info("Environment check passed.")

    # Wrap for SB3 (required even for a single env)
    vec_env = DummyVecEnv([make_env])

    # -----------------------------------------------------------------------
    # 3.  PPO model
    #     MlpPolicy: suitable for the flat observation vector from
    #     ObservationBuilder (rays + agent features + opponent features).
    # -----------------------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        # --- Core hyperparameters (good defaults for Armagetron) ---
        learning_rate=3e-4,
        n_steps=2048,           # steps collected per policy update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,             # discount factor
        gae_lambda=0.95,        # GAE smoothing
        clip_range=0.2,         # PPO clipping
        ent_coef=0.01,          # entropy bonus (encourages exploration)
        vf_coef=0.5,
        max_grad_norm=0.5,
        # --- Logging ---
        tensorboard_log="./tensorboard_logs/",
    )

    # -----------------------------------------------------------------------
    # 4.  Callbacks
    # -----------------------------------------------------------------------
    os.makedirs("./models", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path="./models/",
        name_prefix="armagetron_ppo",
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # 5.  Train
    # -----------------------------------------------------------------------
    TOTAL_TIMESTEPS = 500_000   # increase once the setup is confirmed working

    logger.info("Starting training for %d timesteps…", TOTAL_TIMESTEPS)
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            progress_bar=True,
        )
        model.save("armagetron_ppo_final")
        logger.info("Training complete — model saved as armagetron_ppo_final.zip")

    except KeyboardInterrupt:
        logger.info("Interrupted by user — saving checkpoint…")
        model.save("armagetron_ppo_interrupted")

    finally:
        env.close()
        vec_env.close()


if __name__ == "__main__":
    main()
