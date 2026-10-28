import numpy as np
from gridworld import Gridworld

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


def get_return(state_list,gamma):
    cnt=0
    return_val=0
    for visit in state_list:
        rew= visit[2]
        return_val+= rew*np.power(gamma,cnt)
        cnt+=1
    return return_val

def update_policy(episode_list,policy_matrix,state_action_matrix):

    for visit in episode_list:
        obs=visit[0]
        col=obs[1]+obs[0]*4
        if policy_matrix[obs[0],obs[1]]!=-1:
            policy_matrix[obs[0],obs[1]]=np.argmax(state_action_matrix[:,col])
    return policy_matrix

def main():
    env=Gridworld(3,4)

    state_matrix=np.zeros((3,4))
    state_matrix[1,1]=-1
    state_matrix[1,3]=1
    state_matrix[0,3]=1

    reward_matrix=np.full((3,4),-0.04)
    reward_matrix[0,3]=1
    reward_matrix[1,3]=-1

    policy_matrix=np.random.randint(0,4,size=(3,4)).astype(np.float32)
    policy_matrix[1,1]=np.nan
    policy_matrix[0,3]=policy_matrix[1,3]=-1        

    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    state_action_matrix=np.random.random_sample((4,12))
    running_mean_matrix=np.full((4,12),1.0e-10)
    gamma=0.999
    total_epos=500000
    print_epos=3000

    for epos in range(total_epos):
        episode_list=[]

        obs=env.reset(exploring_starts=True)
        is_starting=True

        for _ in range(1000):
            
            action=policy_matrix[obs[0],obs[1]]

            if is_starting:
                action=np.random.randint(0,4)
                is_starting=False

            new_obs,reward,done=env.step(action)

            episode_list.append((obs,action,reward))
            obs=new_obs
            if done: break

        cnt=0
        checkup_matrix=np.zeros((4,12))
        
        for visit in episode_list:

            obs, action,reward=visit
            row=int(action)
            col=int(obs[1]+obs[0]*4)
            if checkup_matrix[row][col]==0:
                return_val=get_return(episode_list[cnt:],gamma)
                checkup_matrix[row][col]=1
                state_action_matrix[row][col]+=return_val
                running_mean_matrix[row][col]+=1
            cnt+=1

        policy_matrix=update_policy(episode_list,policy_matrix,state_action_matrix/running_mean_matrix)

        if(epos%print_epos)==0:
            print("")
            print("State-Action matrix after " + str(epos+1) + " iterations:") 
            print(state_action_matrix / running_mean_matrix)
            print("Policy matrix after " + str(epos+1) + " iterations:") 
            print(policy_matrix)
            print_policy(policy_matrix)
            

    print("STATE_action matrix after " + str(total_epos) + " iterations:") 
    print(state_action_matrix/running_mean_matrix)

    print_policy(policy_matrix)


            
if __name__== "__main__":
    main()