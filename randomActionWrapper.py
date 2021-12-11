import random 
import gym

class RandomActionWrapper(gym.ActionWrapper):
    
    def action(self, action):
        actions = [0,1,2,3,4,5]
        randomNumber = random.randint(0,9)
        randomWhen = [3,5,6]
        if randomNumber in randomWhen:
            action = random.choice(actions)
        return action

