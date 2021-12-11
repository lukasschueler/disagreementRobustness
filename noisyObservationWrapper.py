import numpy as np
import gym
from gym.spaces import Box

class MakeEnvDynamic(gym.ObservationWrapper):
    """Make observation dynamic by adding noise"""
    def __init__(self, env=None, percentPad=40, action_dependent=False):
        super(MakeEnvDynamic, self).__init__(env)
        self.action_dependent = action_dependent
        self.origShape = env.observation_space.shape
        newside = int(round(max(self.origShape[:-1])*100./(100.-percentPad)))
        self.newShape = [newside, newside, 3]
        self.observation_space = Box(0.0, 255.0, self.newShape)
        self.ob = None
        self.saved_noise = np.random.randint(0,256,self.newShape).astype('uint8')
        self.action_taken = None
        
    def step(self, action):
        self.action_taken = action
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        if self.action_dependent:
            if self.action_taken == 5:
                imNoise = np.random.randint(0,256,self.newShape).astype(obs.dtype)
                self.saved_noise = imNoise
                imNoise[:self.origShape[0], :self.origShape[1], :] = obs[:,:,:]
                self.ob = imNoise
            else:
                self.saved_noise[:self.origShape[0], :self.origShape[1], :] = obs[:,:,:]
                imNoise = self.saved_noise
                self.ob = imNoise
                
        else:
            imNoise = np.random.randint(0,256,self.newShape).astype(obs.dtype)
            imNoise[:self.origShape[0], :self.origShape[1], :] = obs[:,:,:]
            self.ob = imNoise
            
        
        return imNoise