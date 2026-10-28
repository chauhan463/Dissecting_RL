from multi_armed_bandit import MultiArmedBandit
import random
import numpy as np

def return_rmse(predictions,targets):
    return np.sqrt(((predictions-targets)**2).mean())

def boltzmann(x,temperature):
    exponent=np.true_divide(x-np.max(x),temperature)
    return np.exp(exponent)/np.sum(np.exp(exponent))


def return_boltzmann_action(temperature,reward_counter_array):
    tot_actions=reward_counter_array.shape[0]
    boltzmann_distribution= boltzmann(reward_counter_array,temperature)
    action=np.random.choice(tot_actions,p=boltzmann_distribution)
    return action

def main():
    reward_distribution = [0.3, 0.5, 0.8]
    my_bandit= MultiArmedBandit(reward_probability_list=reward_distribution)
    tot_arms=3
    tot_episodes = 2000
    tot_steps = 1000
    temperature_stop=0.1
    temperature_start=0.0001
    
    print_every_episodes = 100
    cumulated_reward_list = list()
    average_utility_array = np.zeros(tot_arms)
    temperature_array=np.linspace(temperature_start,temperature_stop,tot_steps)
    for epis in range(tot_episodes):
        cumulated_reward=0
        reward_counter_array=np.zeros(tot_arms)
        action_counter_array=np.full(tot_arms,1.0e-5)
        for step in range(tot_steps):
            temperature=temperature_array[step]
            action=return_boltzmann_action(temperature,np.true_divide(reward_counter_array,action_counter_array))
            reward=my_bandit.step(action)
            reward_counter_array[action]+=reward
            action_counter_array[action]+=1
            cumulated_reward+=reward
        cumulated_reward_list.append(cumulated_reward)
        utility_array=np.true_divide(reward_counter_array,action_counter_array)
        average_utility_array+=utility_array
        if epis % print_every_episodes == 0:
            print("Episode: " + str(epis))
            print("Cumulated Reward: " + str(cumulated_reward))
            print("Utility distribution: " + str(utility_array))
            print("Utility RMSE: " + str(return_rmse(utility_array, reward_distribution)))
            print("")
            print("")
    print("Average Cumulated Reward: " + str(np.mean(cumulated_reward_list)))
    print("Std Cumulated Reward: " + str(np.std(cumulated_reward_list)))
    print("Average utility distribution: " + str(average_utility_array / tot_episodes))
    print("Average utility RMSE: " + str(return_rmse(average_utility_array/tot_episodes, reward_distribution)))

if __name__ == "__main__":
    main()
                        