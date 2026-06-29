import os
import shutil
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from Lake_environment import LakeMapEnv
from Configuration import SimConfig, set_seed


# -----------------------------
# Settings
# -----------------------------
NUM_ENVS = 20
LOG_DIR = "runs/ppo_lake"
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model.zip")
VECNORM_PATH = os.path.join(LOG_DIR, "vecnorm.1")

TOTAL_TIMESTEPS = int(1e6)   # extra training time
SEED = 42

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
os.makedirs(LOG_DIR, exist_ok=True)
set_seed(SEED)


# -----------------------------
# Environment factory
# -----------------------------
def make_env(rank):
    def thunk():
        env = LakeMapEnv(SimConfig(size=64, obstacle_p=0.01, algae_p=0.00, max_steps=30000))
        env.reset(seed=SEED + rank)
        return env
    return thunk


# -----------------------------
# Optional logging callback
# -----------------------------
class RewardBreakdownCallback(BaseCallback):
    KEYS = (
        "coverage", "r_covered", "r_crash", "r_stall",
        "r_proximity", "r_smoothing", "r_reverse",
        "r_finish", "total_reward"
    )

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.buf = {k: [] for k in self.KEYS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if not info:
                    continue
                for k in self.KEYS:
                    if k in info:
                        self.buf[k].append(info[k])
        return True

    def _on_rollout_end(self) -> None:
        if not any(self.buf[k] for k in self.KEYS):
            return
        means = {k: float(np.mean(self.buf[k])) if self.buf[k] else 0.0 for k in self.KEYS}
        for k, v in means.items():
            self.logger.record(f"rollout/{k}", v)
        if self.verbose:
            print("Reward components (rollout means):", means)
        self.buf = {k: [] for k in self.KEYS}


if __name__ == "__main__":
    # -----------------------------
    # Training env
    # -----------------------------
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    vec_env = VecMonitor(
        vec_env,
        info_keywords=("coverage", "r_covered", "r_crash", "r_stall",
                       "r_proximity", "r_smoothing", "r_reverse",
                       "r_finish", "total_reward")
    )

    vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
    vec_env.training = True
    vec_env.norm_reward = False

    # -----------------------------
    # Eval env
    # -----------------------------
    eval_env = DummyVecEnv([make_env(100)])
    eval_env = VecMonitor(
        eval_env,
        info_keywords=("coverage", "r_covered", "r_crash", "r_stall",
                       "r_proximity", "r_smoothing", "r_reverse",
                       "r_finish", "total_reward")
    )
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # -----------------------------
    # Callbacks
    # -----------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=200000,
        save_path=LOG_DIR,
        name_prefix="checkpoint_resume"
    )

    rb_cb = RewardBreakdownCallback(verbose=1)

    # -----------------------------
    # Load and continue training
    # -----------------------------
    model = PPO.load(
        BEST_MODEL_PATH,
        env=vec_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=LOG_DIR
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_cb, checkpoint_cb, rb_cb],
            reset_num_timesteps=False
        )
    finally:
        model.save(os.path.join(LOG_DIR, "best_model_retrained"))
        vec_env.save(os.path.join(LOG_DIR, "vecnorm_retrained.pkl"))
        vec_env.close()
        eval_env.close()