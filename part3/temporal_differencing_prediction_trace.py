import numpy as np
from gridworld import Gridworld

def update_utility(utility_matrix, alpha,delta ,trace_matrix):

    utility_matrix+= trace_matrix*alpha*delta
    return utility_matrix


def update_trace(trace_matrix, lambda_, gamma):
    trace_matrix*=lambda_*gamma
    return trace_matrix

def main():
    env= Gridworld(3,4)

    state_matrix=np.zeros((3,4))
    state_matrix[0,3]=1
    state_matrix[1,3]=1
    state_matrix[1,1]=-1
    print(state_matrix)

    trace_matrix = np.zeros((3,4))
    print("Trace Matrix:")
    print(trace_matrix)

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
    alpha=0.1
    total_epis=300000
    print_epis=1000
    lambda_=0.5

    utility_matrix=np.zeros((3,4))  


    for epis in range(total_epis):

        observation=env.reset(exploring_starts=True)

        for step in range(1000):

            action= policy_matrix[observation[0],observation[1]]
            new_observation, reward, done= env.step(action)

            delta = reward + gamma*utility_matrix[new_observation[0],new_observation[1]] - utility_matrix[observation[0],observation[1]]

            trace_matrix[observation[0],observation[1]]+=1
            utility_matrix=update_utility(utility_matrix, alpha,delta ,trace_matrix)
            trace_matrix=update_trace(trace_matrix, lambda_, gamma)
            observation=new_observation
            if done:break

            if epis%print_epis==0:
                print("")
                print("Utility matrix after " + str(epis+1) + " iterations:") 
                print(utility_matrix)

    print("")
    print("Utility matrix after " + str(total_epis) + " iterations:") 
    print(utility_matrix)

if __name__ == "__main__":
    main()