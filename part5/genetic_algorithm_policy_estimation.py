from gridworld import Gridworld
import numpy as np


def return_random_population(population_size, chromosome_size, gene_set):
    return np.random.choice(gene_set,size=(population_size,chromosome_size))

def return_best_worst_population(population, fitness_array):
    new_population=np.zeros(population.shape)
    new_fitness_array=np.zeros(fitness_array.shape)
    worst_best_indices= np.argsort(fitness_array)
    best_worst_indices = worst_best_indices[::-1]

    cnt=0
    for idx in best_worst_indices:
        new_population[cnt,:]=population[idx,:]
        new_fitness_array[cnt]=fitness_array[idx]
        cnt+=1
    return new_population,new_fitness_array

def return_truncated_population(population, fitness_array, new_size):
    pop_size=population.shape[0]
    chromosome_size=population.shape[1]
    new_population=np.resize(population,(new_size,chromosome_size))
    new_fitness_array=np.resize(fitness_array,new_size)
    return new_population,new_fitness_array

def return_roulette_selected_population(population, fitness_array, new_size):
    fitness_distribution=np.exp(fitness_array-np.max(fitness_array))/np.sum(np.exp(fitness_array-np.max(fitness_array)))    

    pop_size=population.shape[0]
    chromosome_size=population.shape[1]

    pop_indices=np.random.choice(pop_size,new_size,p=fitness_distribution)

    new_population=np.zeros((new_size,chromosome_size))
    new_fitness_array=np.zeros(new_size)

    cnt=0
    for idx in pop_indices:
        new_population[cnt:]= population[idx,:]
        new_fitness_array[cnt]=fitness_array[idx]
        cnt+=1
    return new_population,new_fitness_array


def return_crossed_population(population, new_size, elite=0):
    pop_size=population.shape[0]
    chromosome_size=population.shape[1]

    if elite>new_size:
        ValueError("Error: elite value can't be greater than new_size")

    new_population= np.zeros((new_size,chromosome_size))
    new_population[0:elite]=population[0:elite]

    parents_index=np.random.randint(low=0,high=pop_size,size=(new_size-elite,2))

    for i in range(elite,new_size-elite):
        first_parent= population[parents_index[i,0],:]
        second_parent= population[parents_index[i,1],:]
        slicing_point=np.random.randint(low=0,high=chromosome_size)
        child=np.zeros(chromosome_size)
        child[0:slicing_point]=first_parent[0:slicing_point]
        child[slicing_point:]=second_parent[slicing_point:]
        new_population[i]=np.copy(child)
    return new_population

def return_mutated_population(population,gene_set,mutation_rate,elite=0):
    for x in np.nditer(population[elite:,:],op_flags=['readwrite']):
        if(np.random.uniform(0,1))<mutation_rate:
            x[...]=np.random.choice(gene_set,1)
    return population


def return_chromosome_string(chromosome_array):
    '''Returns a string where the actions of the chromosome
    are replaced with symbols.

    Attention, this function only works for the 4x3 gridworld
    It must be readapted for larger worlds
    @param chromosome_array
    @return a string of symbols
    '''
    chromosome_string = ""
    counter=0
    for gene in chromosome_array:
        if(counter==3): chromosome_string += ' * '
        elif(counter==7): chromosome_string += ' * '
        elif(counter==5): chromosome_string += ' # '
        else:
            if(gene == 0): chromosome_string += ' ^ '
            elif(gene == 1): chromosome_string += ' > '
            elif(gene == 2): chromosome_string += ' v '
            elif(gene == 3): chromosome_string += ' < '
        counter += 1
    return chromosome_string


def main():
    tot_generations = 100
    tot_episodes = 100
    tot_steps = 14
    population_size = 100
    elite_size = 10
    mutation_rate = 0.10
    gene_set = [0, 1, 2, 3]
    chromosome_size = 12

    world_rows = 3
    world_columns = 4
    env = Gridworld(world_rows, world_columns)


    
    mean_fitness_list = list()
    max_fitness_list = list()
    min_fitness_list = list()

    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    population_matrix=return_random_population(population_size, chromosome_size, gene_set)
    print("Population matrix shape: " + str(population_matrix.shape))

    for generation in range(tot_generations):
        fitness_array=np.zeros((population_size ))

        for chromosome_index in range(population_size):
            for episode in range(tot_episodes):
                observation=env.reset(exploring_starts=True)
                for step in range(tot_steps):
                    col=observation[0]*world_columns+observation[1]
                    action=population_matrix[chromosome_index,:][col]

                    observation, reward , done= env.step(action)
                    fitness_array[chromosome_index]+=reward
                    if done: break

        max_fitness_list.append(np.amax(fitness_array))
        mean_fitness_list.append(np.mean(fitness_array))
        min_fitness_list.append(np.amin(fitness_array))

        print("Generation: " + str(generation+1))
        print("Fitness Mean: " + str(np.mean(fitness_array)))
        print("Fitness STD: " + str(np.std(fitness_array)))
        print("Fitness Max: " + str(np.amax(fitness_array))
              + " at index " + str(np.argmax(fitness_array)))
        print("Fitness Min: " + str(np.amin(fitness_array))
              + " at index " + str(np.argmin(fitness_array)))
        print("Optimal Policy:")
        print(" >  >  >  *  ^  #  ^  *  ^  <  <  <")
        for i in range(int(fitness_array.shape[0]/10)):
            print("Fitness " + str(i) + " ..... " + str(fitness_array[i]))
            print(return_chromosome_string(population_matrix[i,:]))        
        print("")

        population_matrix,fitness_array = return_roulette_selected_population(population_matrix,fitness_array,population_size)
        population_matrix, fitness_array = return_best_worst_population(population_matrix, fitness_array)

        population_matrix = return_crossed_population(population_matrix, 
                                                      population_size, 
                                                      elite=elite_size)
        population_matrix = return_mutated_population(population_matrix,
                                                      gene_set=gene_set,
                                                      mutation_rate=mutation_rate, 
                                                      elite=elite_size)
    try:
        import matplotlib.pyplot as plt

        array=np.arange(1,tot_generations+1,dtype='int32')
        plt.plot(array,mean_fitness_list,color="red",marker="o",markersize=6,markevery=10,label='Mean')
        plt.plot(array,max_fitness_list,color="blue",marker="^",markersize=6,markevery=10,label='Max')
        plt.plot(array,min_fitness_list,color="green",marker="v",markersize=6,markevery=10,label='Min')

        plt.legend(loc="upper center",bbox_to_anchor=(0.5,1.05),ncol=3,fancybox=True, shadow=True)

        plt.ylabel('Fitness',fontsize=15)
        plt.xlabel('Generation',fontsize=15)

        plt.savefig("./fitness.jpg",dpi=500)
    except ImportError as e:
        print("Please install matplotlib")
        pass

if __name__=="__main__":
    main()

                               