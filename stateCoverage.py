import gym
import seaborn as sns
import numpy as np
import wandb
class stateCoverage(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env, envSize, recordWhen, rank):
        super().__init__(env)
        self.envSize = envSize
        self.counts = {}
        self.numberTimesteps = 0
        self.recordWhen = recordWhen
        self.rank = rank
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.rank == 0:
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
                self.createHeatmap(self.counts, self.envSize)

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def createHeatmap(self, dictionary, envSize):
        grid = np.zeros((envSize, envSize))
        for key, value in dictionary.items():
            x = key[0]
            y = key[1]
            grid[x-1][y-1] = value
        heatmap = sns.heatmap(grid)
        wandb.log({"Coverage": [wandb.Image(heatmap)]})