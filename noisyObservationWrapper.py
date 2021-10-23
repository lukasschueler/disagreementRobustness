import numpy as np
import gym
from gym.spaces import Box

class MakeEnvDynamic(gym.ObservationWrapper):
    """Make observation dynamic by adding noise"""
    def __init__(self, env=None, percentPad=5):
        super(MakeEnvDynamic, self).__init__(env)
        
        self.origShape = env.observation_space.shape
        newside = int(round(max(self.origShape[:-1])*100./(100.-percentPad)))
        self.newShape = [newside, newside, 3]
        self.observation_space = Box(0.0, 255.0, self.newShape)
        self.ob = None

    def observation(self, obs):
        imNoise = np.random.randint(0,256,self.newShape).astype(obs.dtype)
        imNoise[:self.origShape[0], :self.origShape[1], :] = obs[:,:,:]
        
        self.ob = imNoise
        return imNoise 