import numpy as np
from mountain_car import MountainCar
import matplotlib.pyplot as plt

def update_state_action(state_action_matrix, visit_counter_matrix,observation,new_observation,action,new_action,reward,alpha,gamma,tot_bins):
    col=observation[0]*tot_bins+observation[1]
    q=state_action_matrix[action,col]
    col_t1=new_observation[0]*tot_bins+new_observation[1]
    q_t1=state_action_matrix[int(new_action),col_t1]
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[action, col])
    state_action_matrix[int(action),col]+=alpha*(reward+gamma*q_t1-q)
    return state_action_matrix




def update_visit_counter(observation,action,visit_counter_matrix,tot_bins):
    col=observation[0]*tot_bins +observation[1]
    visit_counter_matrix[action,col]+=1.0
    return visit_counter_matrix
    


def update_policy(policy_matrix,observation,state_action_matrix,tot_bins):

    col=observation[0]*tot_bins+observation[1]
    best_action=np.argmax(state_action_matrix[:,col])
    policy_matrix[observation[0],observation[1]]=best_action
    return policy_matrix

def print_policy(policy_matrix):
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(policy_matrix[row,col] == 0): policy_string += " <  "
            elif(policy_matrix[row,col] == 1): policy_string += " O  "
            elif(policy_matrix[row,col] == 2): policy_string += " >  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def return_epsilon_greedy_action(policy_matrix, epsilon,observation):
    tot_action=int(np.nanmax(policy_matrix)+1)
    action=int(policy_matrix[observation[0],observation[1]])
    non_greedy_prob=epsilon/tot_action
    weight_array= np.full((tot_action),non_greedy_prob)
    weight_array[action]=1-epsilon+non_greedy_prob
    return np.random.choice(tot_action,p=weight_array)

def return_decayed_value(starting_val,minimum_value,global_step,decay_step):
    decayed_value= starting_val* np.power(0.9,(global_step/decay_step))
    if decayed_value < minimum_value:
        return minimum_value
    else:
        return decayed_value

def plot_curve(data_list, filepath="./my_plot.png",
               x_label="X", y_label="Y",
               x_range=(0, 1), y_range=(0,1), color="-r", kernel_size=50, alpha=0.4, grid=True):
        """Plot a graph using matplotlib

        """
        if(len(data_list) <=1):
            print("[WARNING] the data list is empty, no plot will be saved.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
        ax.grid(grid)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(data_list, color, alpha=alpha)  # The original data is showed in background
        kernel = np.ones(int(kernel_size))/float(kernel_size)  # Smooth the graph using a convolution
        tot_data = len(data_list)
        lower_boundary = int(kernel_size/2.0)
        upper_boundary = int(tot_data-(kernel_size/2.0))
        data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
        #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
        #print("Convolved: " + str(np.arange(tot_data).shape))
        ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary], data_convolved_array, color, alpha=1.0)  # Convolved plot
        fig.savefig(filepath)
        fig.clear()
        plt.close(fig)
        # print(plt.get_fignums())  # print the number of figures opened in background



def main():
    env=MountainCar(mass=0.2,friction=0.3,delta_t=0.1)

    tot_action=3
    tot_bins=12
    velocity_state_array=np.linspace(-1.5,+1.5,num=tot_bins-1,endpoint=False)
    position_state_array=np.linspace(-1.2,+0.5,num=tot_bins-1,endpoint=False)

    policy_matrix=np.random.randint(0,tot_action,(tot_bins,tot_bins)).astype(np.float32)
    print("policy matrix",policy_matrix)

    state_action_matrix=np.zeros((tot_action,tot_bins*tot_bins))
    visit_counter_matrix=np.zeros((tot_action,tot_bins*tot_bins))

    gamma = 0.999
    alpha = 0.001
    tot_episode = 100000
    epsilon_start = 0.9  # those are the values for epsilon decay
    epsilon_stop = 0.1
    epsilon_decay_step = 3000
    print_episode = 500  # print every...
    movie_episode = 10000  # movie saved every...
    reward_list = list()
    step_list = list()

    for epis in range(tot_episode):
        epsilon=return_decayed_value(epsilon_start,epsilon_stop,epis,epsilon_decay_step)

        observation=env.reset(exploring_starts=False)

        observation = (np.digitize(observation[1], velocity_state_array),
                       np.digitize(observation[0], position_state_array))

        is_starting = True
        cumulated_reward=0

        for step in range(100):
            action=return_epsilon_greedy_action(policy_matrix, epsilon,observation)
            if(is_starting):
                action=np.random.randint(0,tot_action)
                is_starting=False

            new_observation, reward, done = env.step(action)
            new_observation = (np.digitize(new_observation[1], velocity_state_array),
                               np.digitize(new_observation[0], position_state_array))
            new_action = policy_matrix[new_observation[0], new_observation[1]]

            state_action_matrix=update_state_action(state_action_matrix, visit_counter_matrix,observation,new_observation,action,new_action,reward,alpha,gamma,tot_bins)

            policy_matrix=update_policy(policy_matrix,observation,state_action_matrix,tot_bins)

            visit_counter_matrix=update_visit_counter(observation,action,visit_counter_matrix,tot_bins)


            observation=new_observation
            cumulated_reward+=reward
            if done: break

        reward_list.append(cumulated_reward)
        step_list.append(step)
        if(epis % print_episode == 0):
            print("")
            print("Episode: " + str(epis+1))
            print("Epsilon: " + str(epsilon))
            print("Episode steps: " + str(step+1))
            print("Cumulated Reward: " + str(cumulated_reward))
            print("Policy matrix: ")
            print_policy(policy_matrix)
        if(epis % movie_episode == 0):
            print("Saving the reward plot in: ./reward.png")
            plot_curve(reward_list, filepath="./reward.png",
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-1.1,1.1),
                       color="red", kernel_size=500,
                       alpha=0.4, grid=True)
            print("Saving the step plot in: ./step.png")
            plot_curve(step_list, filepath="./step.png",
                       x_label="Episode", y_label="Steps",
                       x_range=(0, len(step_list)), y_range=(-0.1,100),
                       color="blue", kernel_size=500,
                       alpha=0.4, grid=True)
            print("Saving the gif in: ./mountain_car.gif")
            env.render(file_path='./mountain_car.gif', mode='gif')
            print("Complete!")


    print("Policy matrix after " + str(tot_episode) + " episodes:")
    print_policy(policy_matrix)
if __name__ == "__main__":
    main()