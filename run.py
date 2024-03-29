#!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
from gym.wrappers import Monitor as VideoMonitor
# from gym.wrappers import FrameStack

import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor as EnvMonitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit, \
    make_unity_maze, StickyActionEnv

import datetime
import wandb
from noisyObservationWrapper import MakeEnvDynamic
from randomActionWrapper import RandomActionWrapper
from stateCoverage import stateCoverage




def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'],
                      num_dyna=args['num_dynamics'],
                      var_output=args['var_output'])
    log, tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process, num_dyna, var_output):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics_class = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet

        # create dynamics list
        self.dynamics_list = []
        for i in range(num_dyna):
            self.dynamics_list.append(self.dynamics_class(auxiliary_task=self.feature_extractor,
                                                          predict_from_pixels=hps['dyn_from_pixels'],
                                                          feat_dim=512, scope='dynamics_{}'.format(i),
                                                          var_output=var_output)
                                      )

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            unity=hps["env_kind"] == "unity",
            dynamics_list=self.dynamics_list
        )

        self.agent.to_report['Feature Extractor Loss'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['Feature Extractor Loss']

        self.agent.to_report['State Predictor Loss'] = tf.reduce_mean(self.dynamics_list[0].partial_loss)
        for i in range(1, num_dyna):
            self.agent.to_report['State Predictor Loss'] += tf.reduce_mean(self.dynamics_list[i].partial_loss)

        self.agent.total_loss += self.agent.to_report['State Predictor Loss']
        
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        if self.hps["env_kind"] == "unity":
            env.close()
            # self.ob_mean, self.ob_std = 124.89177, 55.7459
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics_list=self.dynamics_list)
        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        if args["stickyAtari"]:
            env._max_episode_steps = args['max_episode_steps'] * 4
            env = StickyActionEnv(env)
        else:
            env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        if not args["stickyAtari"]:
            env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'unity':
        env = make_unity_maze(args["env"], seed=args["seed"], rank=rank,
            ext_coeff=args["ext_coeff"], recordUnityVid=args['recordUnityVid'],
            expID=args["unityExpID"], startLoc=args["startLoc"], door=args["door"],
            tv=args["tv"], testenv=args["testenv"], logdir=logger.get_dir())
        
    elif args["env_kind"] == 'custom':
        env = gym.make(args['env'])
        
        time = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f")
        from pathlib import Path
        dataPath = "./disagreeData/ENV" + time
        Path(dataPath).mkdir(parents=True, exist_ok=True)
        
        # env = FrameStack(env, 4)
        env = EnvMonitor(env, dataPath)
        env = VideoMonitor(env, "./disagreeVideo/VID"+ args["exp_name"] + time, video_callable = lambda episode_id: episode_id % args['record_when'] == 0)
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size = args["tile_size"]))
        
        if args["random_actions"]:
            env = RandomActionWrapper(env)        
        if args["add_noise"]:
            env = MakeEnvDynamic(env)        
        if args["record_coverage"]:
            env = stateCoverage(env, args["size"], args["record_when"], rank)        

    # if add_monitor:
    #     env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir='./logs/' +
                                                 datetime.datetime.now().strftime(args["expID"] + "-openai-%Y-%m-%d-%H-%M-%S-%f"),
                                             format_strs=['stdout', 'log',
                                                          'csv', 'tensorboard']
                                             if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='MiniGrid-DoorKey-8x8-v0',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="custom")
    parser.add_argument('--noop_max', type=int, default=30)
    parser.add_argument('--stickyAtari', action='store_true', default=True)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=4)
    


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=8)
    parser.add_argument('--nlumps', type=int, default=1)


def add_unity_params(parser):
    parser.add_argument('--testenv', action='store_true', default=False,
                        help='test mode: slows to real time with bigger screen')
    parser.add_argument('--startLoc', type=int, default=0)
    parser.add_argument('--door', type=int, default=1)
    parser.add_argument('--tv', type=int, default=2)
    parser.add_argument('--unityExpID', type=int, default=0)
    parser.add_argument('--recordUnityVid', action='store_true', default=False)


if __name__ == '__main__':
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_unity_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--expID', type=str, default='000')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--layernorm', type=int, default=0)
    
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--num_dynamics', type=int, default=5)
    parser.add_argument('--var_output', action='store_true', default=True)
    
    parser.add_argument('--exp_name', type=str, default='Just another test')
    parser.add_argument('--ext_coeff', type=float, default=1.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--tile_size', type=int, default=8) # 8 for default, 12 for feature extractor testing
    parser.add_argument('--record_when', type=int, default=400)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--random_actions', default=False)
    parser.add_argument('--record_coverage', default=False)
    parser.add_argument('--add_noise', default=False)
    

    
    # Short runs  
    parser.add_argument('--num_timesteps', type=int, default=1000448)
    # Middle runs
    # parser.add_argument('--num_timesteps', type=int, default=2000000)
    # Long runs  
    # parser.add_argument('--num_timesteps', type=int, default=10000000)


    args = parser.parse_args()
    
    wandb.init(project="thesis", group = "Exploration_by_Disagreement", entity = "lukischueler", name = args.exp_name, config = args)
            #    , monitor_gym = True)
            #    , settings=wandb.Settings(start_method='fork'))
    
    wandb.config.update({"architecture": "disagree"})
    
    # Define the custom x axis metric
    # wandb.define_metric("Number of Episodes")
    wandb.define_metric("Frames seen")
    wandb.define_metric("Number of Updates")

    # Define which metrics to plot against that x-axis
    wandb.define_metric("Episode Reward", step_metric='Number of Updates')
    wandb.define_metric("Length of Episode", step_metric='Number of Updates')
    wandb.define_metric("Recent Best Reward", step_metric='Number of Updates')
    wandb.define_metric("Intrinsic Reward (Batch)", step_metric='Number of Updates')
    wandb.define_metric("Extrinsic Reward (Batch)", step_metric='Number of Updates')
    
    wandb.define_metric("Episode Reward", step_metric='Frames seen')
    wandb.define_metric("Length of Episode", step_metric='Frames seen')
    wandb.define_metric("Recent Best Reward", step_metric='Frames seen')
    
    wandb.define_metric("Intrinsic Reward (Batch)", step_metric='Frames seen')
    wandb.define_metric("Extrinsic Reward (Batch)", step_metric='Frames seen')
    

    start_experiment(**args.__dict__)
