#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy
from baselines.ppo1.pposgd_simple import traj_segment_generator
import tensorflow as tf


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)


def play_model(model_path, environment="Humanoid-v2", seed=0):
    U.make_session(num_cpu=1).__enter__()
    env = make_mujoco_env(environment, seed)

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy

    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(model_path + 'human')

    saver.restore(tf.get_default_session(), model_path)
    # saver.restore(tf.get_default_session(), model_path)
    seg_gen = traj_segment_generator(pi, env, 2048, stochastic=True, render=True)

    seg_gen.__next__()

    env.close()


if __name__ == '__main__':
    play_model("/home/hendrik/PPO/saved_models/Humanoid/human")
