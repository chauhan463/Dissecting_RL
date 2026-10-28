import numpy as np
from drone_landing import DroneLanding
import matplotlib.pyplot as plt

def update_state_action(observation,action,state_action_matrix, visit_counter_matrix,new_observation,gamma,alpha,reward):
    x,y,z=observation
    q_t=state_action_matrix[x,y,z,action]
    x_t1,y_t1,z_t1=new_observation
    q_t1=np.amax(state_action_matrix[x_t1,y_t1,z_t1,:])
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[x,y,z,action])
    state_action_matrix[x,y,z,action]+=alpha*(reward+gamma*q_t1-q_t)
    return state_action_matrix

def update_visit_counter(observation,action,visit_counter_matrix):
    x,y,z=observation
    visit_counter_matrix[x,y,z,action]+=1.0
    return visit_counter_matrix
    


def update_policy(policy_matrix,observation,state_action_matrix):

    x,y,z=observation
    best_action=np.argmax(state_action_matrix[x,y,z,:])
    policy_matrix[x,y,z]=best_action
    return policy_matrix


def return_epsilon_greedy_action(policy_matrix, observation,epsilon=0.1):
    x,y,z=observation
    tot_action=int(np.nanmax(policy_matrix)+1)
    action=int(policy_matrix[x,y,z])
    non_greedy_prob=epsilon/tot_action
    weight_array= np.full((tot_action),non_greedy_prob)
    weight_array[action]=1-epsilon+non_greedy_prob
    return np.random.choice(tot_action,p=weight_array)

def return_decayed_value(starting_val,global_step,decay_step):
    decayed_value= starting_val* np.power(0.1,(global_step/decay_step))
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
    world_size = 21
    env = DroneLanding(world_size)
    tot_actions=6

  

    policy_matrix = np.random.randint(low=0, high=6, size=(world_size,world_size,world_size)).astype(np.float32)

    # Q-table and visit counter
    state_action_matrix = np.zeros((world_size,world_size,world_size,tot_actions))
    visit_counter_matrix = np.zeros((world_size,world_size,world_size,tot_actions))

    # Hyperparameters
    gamma = 0.999
    alpha = 0.001 #constant step size
    tot_episode = 2500000
    print_episode = 1000
    render_episode = 100000
    save_log_episode = 10

    reward_list = list()
  

    


    for episode in range(tot_episode):
        # epsilon=return_decayed_value(0.1,epis,decay_step=50000)
        observation=env.reset(exploring_starts=True)
        epsilon=0.1
        cumulated_reward = 0
        # is_starting=True
        for step in range(50):

            action= return_epsilon_greedy_action(policy_matrix, observation, epsilon=epsilon)
            # if is_starting==True:
            #     action=np.random.randint(0,4)
            #     is_starting=False
            
            new_observation, reward, done= env.step(action)
            
            
            state_action_matrix=update_state_action(observation,action,state_action_matrix,visit_counter_matrix, new_observation,gamma,alpha,reward)
            policy_matrix=update_policy(policy_matrix,observation,state_action_matrix)
            visit_counter_matrix=update_visit_counter(observation,action,visit_counter_matrix)
            observation=new_observation
            cumulated_reward += reward
            if done:break

        if(episode % save_log_episode == 0):
            reward_list.append(cumulated_reward)
        if(episode % print_episode == 0):
            print("")
            print("Episode: " + str(episode) + " of " + str(tot_episode))
            print("Epsilon: " + str(epsilon))
            print("Cumulated reward: " + str(cumulated_reward))
            print("Q-max: " + str(np.amax(state_action_matrix)))
            print("Q-mean: " + str(np.mean(state_action_matrix)))
            print("Q-min: " + str(np.amin(state_action_matrix)))
        if episode % render_episode == 0:
            print("Saving the gif in ./drone_landing.gif")
            env.render(file_path='./drone_landing.gif', mode='gif')
            print("Done!")
            print("Saving the reward graph in ./reward.png")
            plot_curve(reward_list, filepath="./reward.png", 
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-1.55,1.05),
                       color="red", kernel_size=500, 
                       alpha=0.4, grid=True)
            print("Done!")
    #Training complete
    print("Finished!!!")


if __name__ == "__main__":
    main()