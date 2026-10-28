import numpy as np

class MultiArmedBandit:
    def __init__(self,reward_probability_list=[0.3,0.5,0.8]):
        self.reward_probability_list =reward_probability_list

    def step(self,action):
        if action> len(self.reward_probability_list):
            raise Exception("The action is out of range of bandit.")
        p=self.reward_probability_list[action]
        f=1.0-p
        return np.random.choice(2,p=[p,f])
        