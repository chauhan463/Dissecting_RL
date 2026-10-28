import numpy as np


def return_state_utility(v,u,T,reward,gamma):
    action_array=np.zeros(4)
    for action in range(4):
        action_array[action]= np.sum(np.multiply(u, np.dot(v,T[:,:,action])))
    return reward + gamma*np.max(action_array)
        




def main():

    T=np.load('T.npy')
    tot_states=12
    gamma = 0.99999999999 
    iteration = 0 
    epsilon = 0.001 

    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                    -0.04,   0.0, -0.04,  -1.0,
                    -0.04, -0.04, -0.04, -0.04])    

    new_u = np.zeros(12)

    while True:
        iteration+=1
        curr_u=new_u.copy()
        for s in range(tot_states):
            v=np.zeros((1,tot_states))
            v[0,s]=1.0
            reward=r[s]
            new_u[s]=return_state_utility(v,curr_u,T,reward,gamma)
            delta= np.abs(curr_u[s]-new_u[s])
        if delta < epsilon*(1-gamma)/gamma:
            print("=================== FINAL RESULT ==================")
            print("Iterations: " + str(iteration))
            print("Delta: " + str(delta))
            print("Gamma: " + str(gamma))
            print("Epsilon: " + str(epsilon))
            print("===================================================")
            print(curr_u[0:4])
            print(curr_u[4:8])
            print(curr_u[8:12])
            print("===================================================")
            break


 
if __name__ == "__main__":
    main()