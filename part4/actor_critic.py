import numpy as np
from gridworld import Gridworld

def softmax(x):
    return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))


def update_critic(utility_matrix,observation,new_observation,reward, alpha, gamma):
    u_t=utility_matrix[observation[0],observation[1]]
    u_t1=utility_matrix[new_observation[0],new_observation[1]]
    delta = reward + gamma*u_t1 - u_t
    utility_matrix[observation[0],observation[1]]+=alpha*delta
    return utility_matrix,delta

def update_actor(state_action_matrix,observation,action,delta,beta_matrix=None):
    col=observation[0]*4+observation[1]

    if beta_matrix==None:
        beta=1
    else:
        beta=beta_matrix[action,col]
    state_action_matrix[action,col]+=beta*delta
    return state_action_matrix


def main():


    env = Gridworld(3,4)
    state_matrix=np.zeros((3,4))
    state_matrix[1,1]= -1
    state_matrix[0,3]=1
    state_matrix[1,3]=1


    reward_matrix=np.full((3,4),-0.04)
    reward_matrix[0,3]=1
    reward_matrix[1,3]=-1

    transition_matrix=np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    state_action_matrix=np.random.random((4,12))

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix=np.zeros((3,4))

    gamma = 0.999
    alpha = 0.001 #constant step size
    beta_matrix = np.zeros((4,12))
    tot_epoch = 300000
    print_epoch = 1000

    for epoch in range(tot_epoch):
        observation=env.reset(exploring_starts=True)
        for step in range(1000):
            col=observation[0]*4+observation[1]
            action_array=state_action_matrix[:,col]

            action_distribution=softmax(action_array)
            action=np.random.choice(4,p=action_distribution)

            # beta_matrix[action,col]+=1

            new_observation,reward,done=env.step(action)
            utility_matrix,delta=update_critic(utility_matrix,observation,new_observation,reward, alpha, gamma)
            state_action_matrix=update_actor(state_action_matrix,observation,action,delta,beta_matrix=None)
            if done: break

        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(utility_matrix)
            print("")
            print("State-Action matrix after " + str(epoch+1) + " iterations:") 
            print(state_action_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility_matrix)
    print("State-Action matrix after  " + str(tot_epoch) + " iterations:")
    print(state_action_matrix)



if __name__=="__main__":
    main()
