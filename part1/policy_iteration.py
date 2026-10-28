import numpy as np


def return_expected_action(v,u,T):
    action_array=np.zeros(4)
    for action in range(4):
        action_array[action]= np.sum(np.multiply(u, np.dot(v,T[:,:,action])))
    return np.argmax(action_array)




def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v=np.zeros((1,12))
            v[0,s]=1.0
            action=int(p[s])
            u[s]=r[s]+gamma*np.sum(np.multiply(u, np.dot(v,T[:,:,action])))
    return u


def print_policy(p, shape):
    print("--------------------------")
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "           
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def main():
    gamma = 0.999
    epsilon = 0.0001
    iteration = 0
    T = np.load("T.npy")

    p=np.random.randint(0,4,size=(12)).astype(np.float32)
    p[5] = np.nan
    p[3] = p[7] = -1
    u=np.zeros(12)

    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                -0.04,   0.0, -0.04,  -1.0,
                -0.04, -0.04, -0.04, -0.04])  

    while True:
        iteration+=1
        u1=u.copy()
        u=return_policy_evaluation(p, u, r, T, gamma)
        delta=np.absolute(u-u1).max()
        if delta<(epsilon*(1-gamma)/gamma):
            break


        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v=np.zeros((1,12))
                v[0,s]=1.0
                action=return_expected_action(v,u,T)
                if action!=p[s]:
                    p[s]=action

        print_policy(p,shape=(3,4))
    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")









if __name__ == "__main__":
    main()