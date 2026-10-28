import numpy as np
from gridworld import Gridworld


def get_return(gamma,state_list):
    cnt=0
    rtn= 0
    for obs, reward in state_list:
        rtn+= reward*np.power(gamma,cnt)
        cnt+=1
    return rtn


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

    policy_matrix = np.array([[1,      1,  1,  -1],
                              [0, np.nan,  0,  -1],
                              [0,      3,  3,   3]])
    print("Policy Matrix:")
    print(policy_matrix)

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)


    gamma=0.999 
    total_epis=5000
    print_epis=1000

    utility_matrix=np.zeros((3,4))

    running_mean_matrix=np.full((3,4),1.0e-10)


    for i in range(total_epis):

        episode_list=list()
  
        observation=env.reset(exploring_starts=False)
  
        for j in range(1000):
            x,y=observation
            
            action=policy_matrix[x,y]
          
            observation, reward, done =env.step(action)

            episode_list.append((observation,reward))
            if done:
                break

        checkup_matrix=np.zeros((3,4))

        cnt=0
    
        for obs,rew in episode_list:
            x,y=obs
            if checkup_matrix[x,y]==0:
                return_value = get_return(gamma,episode_list[cnt:])
                utility_matrix[x,y]+=return_value
                running_mean_matrix[x,y]+=1
                checkup_matrix[x,y]=1
            cnt+=1

        if i%print_epis==0:
            print("utility_matrix after "+ str(i+1) +"iterations")
            print(utility_matrix/running_mean_matrix)

    print("utility_matrix after "+ str(total_epis) +"iterations")
    print(utility_matrix/running_mean_matrix)





            

main()
