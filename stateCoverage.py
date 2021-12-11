import gym
import seaborn as sns
import wandb
import numpy as np

class stateCoverage(gym.core.Wrapper):

    def __init__(self, env, envSize=8, recordWhen=10):
        super().__init__(env)
        self.envSize = envSize
        self.counts = {}
        self.numberTimesteps = 0
        self.recordWhen = recordWhen
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.numberTimesteps += 1

        if action == 2:
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
            grid = np.zeros((self.envSize, self.envSize))
            for key, value in self.counts.items():
                x = key[0]
                y = key[1]
                grid[y][x] = value
            grid_cropped = grid[1:-1,1:-1]
            svm= sns.heatmap(grid_cropped)
            # figure = svm.get_figure()    
            # figure.savefig('svm_conf.png', dpi=400)
            wandb.log({"Coverage": [wandb.Image(svm)]})

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)