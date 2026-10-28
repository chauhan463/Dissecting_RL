import numpy as np
from gridworld import Gridworld

def update_state_action(observation,action,state_action_matrix, visit_counter_matrix,new_observation,new_action,gamma,alpha,reward):
    col_t=(observation[0]*4)+observation[1]
    q_t=state_action_matrix[int(action),col_t]

    col_t1=(new_observation[0]*4)+new_observation[1]
    q_t1=state_action_matrix[int(new_action),col_t1]
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[action, col_t])
    state_action_matrix[int(action),col_t]+=alpha*(reward+gamma*q_t1-q_t)
    return state_action_matrix

def update_visit_counter(observation,action,visit_counter_matrix):
    col=observation[0]*4 +observation[1]
    visit_counter_matrix[action,col]+=1.0
    return visit_counter_matrix
    


def update_policy(policy_matrix,observation,state_action_matrix):

    col=observation[0]*4+observation[1]
    best_action=np.argmax(state_action_matrix[:,col])
    policy_matrix[observation[0],observation[1]]=best_action
    return policy_matrix

def print_policy(policy_matrix):
    shape=np.shape(policy_matrix)
    policy_str=""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if policy_matrix[row][col]==-1: policy_str+="*"
            elif policy_matrix[row][col]==0: policy_str+="^"
            elif policy_matrix[row][col]==1: policy_str+=">"
            elif policy_matrix[row][col]==2: policy_str+="v"
            elif policy_matrix[row][col]==3: policy_str+="<"
            elif np.isnan(policy_matrix[row][col]): policy_str+="#"
        policy_str+="\n"
    print(policy_str)

def return_epsilon_greedy_action(policy_matrix, epsilon,observation):
    tot_action=int(np.nanmax(policy_matrix)+1)
    action=int(policy_matrix[observation[0],observation[1]])
    non_greedy_prob=epsilon/tot_action
    weight_array= np.full((tot_action),non_greedy_prob)
    weight_array[action]=1-epsilon+non_greedy_prob
    return np.random.choice(tot_action,p=weight_array)

def return_decayed_value(starting_val,global_step,decay_step):
    decayed_value= starting_val* np.power(0.1,(global_step/decay_step))
    return decayed_value


def main():
    env= Gridworld(3,4)

    state_matrix=np.zeros((3,4))
    state_matrix[0,3]=1
    state_matrix[1,3]=1
    state_matrix[1,1]=-1
    print(state_matrix)

    reward_matrix=np.full((3,4),-0.04)
    reward_matrix[0,3]=1
    reward_matrix[1,3]=-1
    print(reward_matrix)

    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    policy_matrix=np.random.randint(0,4,size=(3,4)).astype(np.float32)
    policy_matrix[1,1]=np.nan
    policy_matrix[0,3]=policy_matrix[1,3]=-1        


    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)


    state_action_matrix=np.zeros((4,12))
    visit_counter_matrix=np.zeros((4,12))

    gamma=0.999 
    alpha=0.05
    total_epis=180000
    print_epis=1000
  

    


    for epis in range(total_epis):
        epsilon=return_decayed_value(0.1,epis,decay_step=100000)
        observation=env.reset(exploring_starts=True)
        is_starting=True
        for step in range(1000):

            action= return_epsilon_greedy_action(policy_matrix, epsilon=0.1, observation=observation)
            if is_starting==True:
                action=np.random.randint(0,4)
                is_starting=False
            
            new_observation, reward, done= env.step(action)
            
            new_action=policy_matrix[new_observation[0],new_observation[1]]
            
            state_action_matrix=update_state_action(observation,action,state_action_matrix,visit_counter_matrix, new_observation,new_action,gamma,alpha,reward)
            policy_matrix=update_policy(policy_matrix,observation,state_action_matrix)
            visit_counter_matrix=update_visit_counter(observation,action,visit_counter_matrix)
            observation=new_observation
            if done:break

        if epis%print_epis==0:
            print("")
            print("policy matrix after " + str(epis+1) + " iterations:") 
            print(policy_matrix)
            print_policy(policy_matrix)

    print("")
    print("Policy matrix after " + str(total_epis) + " iterations:") 
    print(policy_matrix)
    print_policy(policy_matrix)

if __name__ == "__main__":
    main()