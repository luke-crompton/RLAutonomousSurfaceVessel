import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import LinearSchedule

from Lake_environment import LakeMapEnv
from Configuration import SimConfig, set_seed
import shutil


NUM_ENVS = 20
LOG_DIR = "runs/ppo_lake"
RESET_ALL = False
if RESET_ALL and os.path.isdir(LOG_DIR):
    shutil.rmtree(LOG_DIR)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
os.makedirs(LOG_DIR, exist_ok=True)
#repeatable
set_seed(42)

print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")


# make environment
def make_env(rank):
    #returns a new environment rank lets you do a different seed per env
    def thunk():
        env = LakeMapEnv(SimConfig(size=64, obstacle_p=0.01, algae_p=0.00, max_steps=30000))
        env.reset(seed = 42+rank)
        return env
    return thunk


class RewardBreakdownCallback(BaseCallback):
    KEYS = ("coverage", "r_covered", "r_crash", "r_stall", "r_proximity",
             "r_smoothing","r_reverse","r_finish", "total_reward")
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.buf = {k: [] for k in self.KEYS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            # infos is a list (one per env) for SubprocVecEnv
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
            self.logger.record(f"rollout/{k}", v)  # shows up in TensorBoard
        if self.verbose:
            print("Reward components (rollout means):", means)
        self.buf = {k: [] for k in self.KEYS}
        
        
        

if __name__ == "__main__":
    #runs in separate processes
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    vec_env = VecMonitor(vec_env, info_keywords=( "coverage","r_covered", "r_crash", "r_stall", "r_proximity",
                                                  "r_smoothing", "total_reward") )
    vec_env = VecNormalize(vec_env, norm_obs= True, norm_reward=False, clip_obs=10.0)
    #defining eval env to be an environment with size 64
    eval_env = DummyVecEnv([make_env(1)])
    eval_env = VecMonitor(eval_env, info_keywords= ("coverage", "r_covered", "r_crash", "r_stall", "r_proximity",
                                                   "r_smoothing", "total_reward") )
    eval_env = VecNormalize(eval_env, norm_obs= True, norm_reward=False, clip_obs=10.0)

    eval_env.obs_rms = vec_env.obs_rms
    eval_env.training = False
    eval_env.norm_rewards = False

    #evaluating performance every 20,000 training steps
    eval_cb = EvalCallback(eval_env,best_model_save_path= LOG_DIR, log_path= LOG_DIR, eval_freq=50000,
                           n_eval_episodes=10, deterministic=True)

    #saving checkpoints for safety
    callback_point = CheckpointCallback(save_freq=200000, save_path=LOG_DIR, name_prefix="checkpoint")
    model = PPO(
        "MlpPolicy", vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        learning_rate=LinearSchedule(1e-4, 5e-5, 1),
        gamma=0.9999,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        target_kl=0.02,
        sde_sample_freq=64,
        policy_kwargs=dict(net_arch=[64, 64], ortho_init=True),
    )


    rb = RewardBreakdownCallback(verbose=1)
    try:
        model.learn(total_timesteps=int(1e6), callback=[eval_cb, callback_point, rb])
        
    finally:
        print("saving model")
        model.save(os.path.join(LOG_DIR, "Final model"))
        vec_env.save(os.path.join(LOG_DIR,"vecnorm.1"))
        vec_env.close()

