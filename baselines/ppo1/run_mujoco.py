#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import tensorflow as tf


def train(env_id, num_iters, seed, save):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
                        max_iters=num_iters, save=save,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                        gamma=0.99, lam=0.95, schedule='linear'
                        )
    env.close()


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_iters=args.num_iters, seed=args.seed, save=args.save_model)


if __name__ == '__main__':
    main()
