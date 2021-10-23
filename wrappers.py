import os
import itertools
from collections import deque
from copy import copy

import gym
from gym.spaces import Box
import wandb
import seaborn as sns
import numpy as np
from PIL import Image
from utils import save_np_as_mp4


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        elif frame.size == 84 * 84 * 3:  # unity maze
            img = frame
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            x_t = np.reshape(img, [84, 84, 1])
            return x_t.astype(np.uint8)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size,
                                                              resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if self.random_state_copy is not None:
            info['random_state'] = self.random_state_copy
            self.random_state_copy = None
        return ob, r, d, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.random_state_copy = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    ram_map = {
        "room": dict(
            index=3,
        ),
        "x": dict(
            index=42,
        ),
        "y": dict(
            index=43,
        ),
    }

    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.visited = set()
        self.visited_rooms = set()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ram_state = unwrap(self.env).ale.getRAM()
        for name, properties in MontezumaInfoWrapper.ram_map.items():
            info[name] = ram_state[properties['index']]
        pos = (info['x'], info['y'], info['room'])
        self.visited.add(pos)
        self.visited_rooms.add(info["room"])
        if done:
            info['mz_episode'] = dict(pos_count=len(self.visited),
                                      visited_rooms=copy(self.visited_rooms))
            self.visited.clear()
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class MarioXReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.

    def reset(self):
        ob = self.env.reset()
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = \
            info["levelLo"], info["levelHi"], info["xscrollHi"], info["xscrollLo"]
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
            self.visited_levels.add(tuple(self.current_level))
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))
        return ob, reward, done, info


class UnityRoomCounterWrapper(gym.Wrapper):
    def __init__(self, env,use_ext_reward=True):
        gym.Wrapper.__init__(self, env)
        self.current_room = None
        self.visited_rooms = set()
        self.use_ext_reward = use_ext_reward

    def reset(self):
        ob = self.env.reset()
        self.current_room = None
        self.visited_rooms = set()
        return ob


    def step(self, action):
        ob, true_reward, done, info = self.env.step(action)
        reward = 0.0
        current_room = info["curRoom"]

        if self.current_room is None:
            self.current_room = current_room
            reward = 1.0
            self.visited_rooms.add(self.current_room)

        if current_room != self.current_room:
            self.current_room = current_room
            if self.current_room not in self.visited_rooms:
                reward = 1.0
                self.visited_rooms.add(self.current_room)
            else:
                reward = 0.0
        info = {"unity_rooms":copy(self.visited_rooms)}
        return ob, reward if not self.use_ext_reward else true_reward, done, info


class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info


def make_mario_env(crop=True, frame_stack=True, clip_rewards=False):
    assert clip_rewards is False
    import gym
    import retro
    from baselines.common.atari_wrappers import FrameStack

    #gym.undo_logger_setup()
    env = retro.make('SuperMarioBros-Nes', 'Level1-1')
    buttons = env.buttons
    env = MarioXReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=crop)
    if frame_stack:
        env = FrameStack(env, 4)
    env = LimitedDiscreteActions(env, buttons)
    return env


class OneChannel(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(OneChannel, self).__init__(env)
        assert env.observation_space.dtype == np.uint8
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return obs[:, :, 2:3]


class RetroALEActions(gym.ActionWrapper):
    def __init__(self, env, all_buttons, n_players=1):
        gym.ActionWrapper.__init__(self, env)
        self.n_players = n_players
        self._num_buttons = len(all_buttons)
        bs = [-1, 0, 4, 5, 6, 7]
        actions = []

        def update_actions(old_actions, offset=0):
            actions = []
            for b in old_actions:
                for button in bs:
                    action = []
                    action.extend(b)
                    if button != -1:
                        action.append(button + offset)
                    actions.append(action)
            return actions

        current_actions = [[]]
        for i in range(self.n_players):
            current_actions = update_actions(current_actions, i * self._num_buttons)
        self._actions = current_actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons * self.n_players)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class NoReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        return ob, 0.0, done, info


def make_multi_pong(frame_stack=True):
    import gym
    import retro
    from baselines.common.atari_wrappers import FrameStack
    gym.undo_logger_setup()
    game_env = env = retro.make('Pong-Atari2600', players=2)
    env = RetroALEActions(env, game_env.BUTTONS, n_players=2)
    env = NoReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=False)
    if frame_stack:
        env = FrameStack(env, 4)

    return env


def make_unity_maze(env_id, seed=0, rank=0, expID=0, frame_stack=True,
        logdir=None, ext_coeff=1.0, recordUnityVid=False, **kwargs):
    import os
    import sys
    import time
    try:
        sys.path.insert(0, os.path.abspath("ml-agents/python/"))
        from unityagents import UnityEnvironment
        from unity_wrapper import GymWrapper
    except ImportError:
        print("Import error in unity environment. Ignore if not using unity.")
        pass
    from baselines.common.atari_wrappers import FrameStack
    # gym.undo_logger_setup()  # deprecated in new version of gym

    # max 20 workers per expID, max 30 experiments per machine
    if rank>=0 and rank<=200:
        time.sleep(rank * 2)
    env = UnityEnvironment(file_name='envs/' + env_id,
        worker_id=(expID % 60) * 200 + rank)
    maxsteps = 3000 if 'big' in env_id else 500
    env = GymWrapper(env, seed=seed, rank=rank, expID=expID, maxsteps=maxsteps,
        **kwargs)
    if "big" in env_id:
        env = UnityRoomCounterWrapper(env, use_ext_reward=(ext_coeff != 0.0))
    if rank == 1 and recordUnityVid:
        env = RecordBestScores(env, directory=logdir, freq=1)
    print('Loaded environment %s with rank %d\n\n' % (env_id, rank))

    # env = NoReward(env)
    # env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=False)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class RecordBestScores(gym.Wrapper):
    def __init__(self, env, directory, freq=100):
        super(RecordBestScores, self).__init__(env)
        self.freq = freq
        self.frames = []
        self.highest_reward = None
        self.episodic_reward = 0.
        self.longest_length = 0
        self.directory = directory
        self.episode_number = 0
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def _step(self, action):
        state, reward, done, info = self.env.step(action)
        self.frames.append(self.env.render(mode='rgb_array'))
        self.episodic_reward += reward
        if done:
            if self.highest_reward == None:
                self.highest_reward = self.episodic_reward
                self._record_last_episode("high_score_")
            elif self.highest_reward < self.episodic_reward:
                self.highest_reward = self.episodic_reward
                self._record_last_episode("high_score_")
            elif self.episode_number % self.freq == 0:
                self._record_last_episode("random_")

            self.frames = []
            self.episodic_reward = 0
            self.episode_number += 1
        return state, reward, done, info

    def _record_last_episode(self, prefix=""):
        save_np_as_mp4(self.frames, os.path.join(self.directory, prefix+'replay{}.mp4'.format(self.episode_number)))


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class OneHotPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.spaces['image'].shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {
            'mission': obs['mission'],
            'image': out
        }

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class ViewSizeWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

from .minigrid import Goal
class DirectionObsWrapper(gym.core.ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """
    def __init__(self, env,type='slope'):
        super().__init__(env)
        self.goal_position = None
        self.type = type

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)
        return obs

    def observation(self, obs):
        slope = np.divide( self.goal_position[1] - self.agent_pos[1] ,  self.goal_position[0] - self.agent_pos[0])
        obs['goal_direction'] = np.arctan( slope ) if self.type == 'angle' else slope
        return obs

class StateCoverage(gym.core.Wrapper):
    
    
    """
    Text
    """

    def __init__(self, env, envSize, recordWhen):
        super().__init__(env)
        self.envSize = envSize
        self.counts = {}
        self.numberTimesteps = 0
        self.recordWhen = recordWhen
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.numberTimesteps += 1

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        if self.numberTimesteps % self.recordWhen == 0:
            self.createHeatMap(self)

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

    def createHeatmap(self):
        grid = np.zeros((self.envSize, self.envSize))
        for key, value in self.counts.items():
            x = key[0]
            y = key[1]
            grid[x-1][y-1] = value
        heatmap = sns.heatmap(grid)
        wandb.log({"Coverage": [wandb.Image(heatmap)]})
        
class MakeEnvDynamic(gym.ObservationWrapper):
    """Make observation dynamic by adding noise"""
    def __init__(self, env=None, percentPad=5):
        super(MakeEnvDynamic, self).__init__(env)
        self.origShape = env.observation_space.shape
        newside = int(round(max(self.origShape[:-1])*100./(100.-percentPad)))
        self.newShape = [newside, newside, 3]
        self.observation_space = Box(0.0, 255.0, self.newShape)
        self.bottomIgnore = 20  # doom 20px bottom is useless
        self.ob = None

    def _observation(self, obs):
        imNoise = np.random.randint(0,256,self.newShape).astype(obs.dtype)
        imNoise[:self.origShape[0]-self.bottomIgnore, :self.origShape[1], :] = obs[:-self.bottomIgnore,:,:]
        self.ob = imNoise
        return imNoise