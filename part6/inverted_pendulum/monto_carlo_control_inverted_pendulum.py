import numpy as np
from inverted_pendulum import InvertedPendulum
import matplotlib.pyplot as plt

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


def get_return(state_list,gamma):
    cnt=0
    return_val=0
    for visit in state_list:
        rew= visit[2]
        return_val+= rew*np.power(gamma,cnt)
        cnt+=1
    return return_val

def return_decayed_value(starting_val,minimum_value,global_step,decay_step):
    decayed_value= starting_val* np.power(0.9,(global_step/decay_step))
    if decayed_value < minimum_value:
        return minimum_value
    else:
        return decayed_value
    
def update_policy(episode_list,policy_matrix,state_action_matrix,tot_bins):

    for visit in episode_list:
        obs=visit[0]
        col=obs[1]+obs[0]*tot_bins
        if policy_matrix[obs[0],obs[1]]!=-1:
            policy_matrix[obs[0],obs[1]]=np.argmax(state_action_matrix[:,col])
    return policy_matrix

def return_epsilon_greedy_action(policy_matrix,observation, epsilon=0.1):
    tot_action=int(np.nanmax(policy_matrix)+1)
    action=int(policy_matrix[observation[0],observation[1]])
    non_greedy_prob=epsilon/tot_action
    weight_array= np.full((tot_action),non_greedy_prob)
    weight_array[action]=1-epsilon+non_greedy_prob
    return np.random.choice(tot_action,p=weight_array)

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
    env=InvertedPendulum(pole_mass=2.0, cart_mass=8.0, pole_length=0.5, delta_t=0.1)

    tot_action = 3  # Three possible actions
    tot_bins = 12  # the value used to discretize the space
    velocity_state_array = np.linspace(-np.pi, np.pi, num=tot_bins-1, endpoint=False)
    position_state_array = np.linspace(-np.pi/2.0, np.pi/2.0, num=tot_bins-1, endpoint=False)

    #Random policy
    policy_matrix = np.random.randint(low=0, high=tot_action, size=(tot_bins,tot_bins))
    print("Policy Matrix:")
    print_policy(policy_matrix)

    state_action_matrix = np.zeros((tot_action, tot_bins*tot_bins))
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((tot_action, tot_bins*tot_bins), 1.0e-10) 
    gamma = 0.999
    tot_episode = 500000 # 500k
    epsilon_start = 0.99  # those are the values for epsilon decay
    epsilon_stop = 0.1
    epsilon_decay_step = 10000
    print_episode = 500  # print every...
    movie_episode = 20000  # movie saved every...
    reward_list = list()
    step_list = list()


    for episode in range(tot_episode):
        epsilon = return_decayed_value(epsilon_start, epsilon_stop, episode, decay_step=epsilon_decay_step)
        episode_list=[]

        observation=env.reset(exploring_starts=True)
        observation = (np.digitize(observation[1],velocity_state_array),np.digitize(observation[0],position_state_array))
        is_starting=True
        cumulated_reward=0
        for step in range(1000):
            
            action=return_epsilon_greedy_action(policy_matrix,observation, epsilon)

            if is_starting:
                action=np.random.randint(0,tot_action)
                is_starting=False

            new_observation,reward,done=env.step(action)
            new_observation = (np.digitize(new_observation[1], velocity_state_array), 
                               np.digitize(new_observation[0], position_state_array))  
            episode_list.append((observation,action,reward))
            observation=new_observation
            cumulated_reward+=reward
            if done: break

        cnt=0
        checkup_matrix = np.zeros((tot_action, tot_bins*tot_bins))
        
        for visit in episode_list:

            observation, action,reward=visit
            row=int(action)
            col=int(observation[1]+observation[0]*tot_bins)
            if checkup_matrix[row][col]==0:
                return_val=get_return(episode_list[cnt:],gamma)
                checkup_matrix[row][col]=1
                state_action_matrix[row][col]+=return_val
                running_mean_matrix[row][col]+=1
            cnt+=1

        policy_matrix=update_policy(episode_list,policy_matrix,state_action_matrix/running_mean_matrix,tot_bins)

        reward_list.append(cumulated_reward)
        step_list.append(step)
        # Printing utilities
        if(episode % print_episode == 0):
            print("")
            print("Episode: " + str(episode+1))
            print("Epsilon: " + str(epsilon))
            print("Episode steps: " + str(step+1))
            print("Cumulated Reward: " + str(cumulated_reward))
            print("Policy matrix: ") 
            print_policy(policy_matrix)
        if(episode % movie_episode == 0):
            print("Saving the reward plot in: ./reward.png")
            plot_curve(reward_list, filepath="./reward.png", 
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-0.1,100),
                       color="red", kernel_size=500, 
                       alpha=0.4, grid=True)
            print("Saving the step plot in: ./step.png")
            plot_curve(step_list, filepath="./step.png", 
                       x_label="Episode", y_label="Steps", 
                       x_range=(0, len(step_list)), y_range=(-0.1,100),
                       color="blue", kernel_size=500, 
                       alpha=0.4, grid=True)
            print("Saving the gif in: ./inverted_pendulum.gif")
            env.render(file_path='./inverted_pendulum.gif', mode='gif')
            print("Complete!")

    print("Policy matrix after " + str(tot_episode) + " episodes:")
    print_policy(policy_matrix)


            
if __name__== "__main__":
    main()