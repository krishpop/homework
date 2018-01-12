#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import utils

from sklearn.model_selection import train_test_split

save_path = './expert_data'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--save_data', action='store_true',
                        help='saves data to expert_data/envname.npz if True')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    if args.save_data:
        train_obs, test_obs, train_act, test_act = train_test_split(
            np.array(observations), np.array(actions).reshape((len(actions), len(action[0])))
        )
        expert_data = {'_data': train_obs,
                       '_labels': train_act,
                       '_test_data': test_obs,
                       '_test_labels': test_act,
                       '_envname': args.envname}
        expert_data = utils.DataSet(**expert_data)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        expert_data.save(os.path.join(SAVE_PATH, args.envname + '.npz'))

if __name__ == '__main__':
    main()
