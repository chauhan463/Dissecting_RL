import numpy as np
from gridworld import Gridworld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm
from MLP import MLP



def init_env():
    env = Gridworld(5, 5)
  
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])

    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])

    reward_matrix = np.array([[1.0, 0.0, 0.0, 0.0, -1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0, 1.0]])

    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    return env

def update(my_mlp, new_observation, reward, learning_rate, gamma, done):
    if done:
        x= np.array(new_observation,dtype=np.float32)
        target=np.array([reward],dtype=np.float32)
        error=my_mlp.train(x,target,learning_rate)
    else:
        x = np.array(new_observation, dtype=np.float32)  
        target = np.array((reward + (gamma * my_mlp.forward(x))), dtype=np.float32)
        error = my_mlp.train(x, target, learning_rate)  
    return my_mlp,error


def print_utility(my_mlp, tot_rows, tot_cols, decimal=2, flip=True):
    utility_matrix = np.zeros((tot_rows, tot_cols))
    for row in range(tot_rows):
        for col in range(tot_cols):
            x = np.array([row, col], dtype=np.float32)
            utility_matrix[row,col] = my_mlp.forward(x)[0]
    np.set_printoptions(precision=decimal) #set print precision of numpy
    if flip:
        print(np.flipud(utility_matrix))
    else:
        print(utility_matrix)
    np.set_printoptions(precision=8) #reset to default

def subplot(my_mlp, world_size, filename="figure.png"):
            #Define the main figure property
            fig, ax = plt.subplots(nrows=1, ncols=4, subplot_kw={'projection': '3d', 'autoscale_on':False, 'aspect':'equal'})
            #XOR color
            color_00 = "green"
            color_11 = "green"
            color_10 = "red"
            color_01 = "red"

            #Quadratic subplot
            ax[0].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[0].set_xticks(np.arange(0, world_size+1, 1))
            ax[0].set_xticklabels('', fontsize=10)
            ax[0].set_yticklabels('', fontsize=10)
            ax[0].set_yticks(np.arange(0, world_size+1, 1))
            ax[0].set_zlim(-1.0,1.0)
            #ax[0].set_zticklabels(['-1.0','','0','','1.0'], fontsize=10)
            ax[0].view_init(elev=30, azim=-115)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))            
            grid = np.arange(0.0, world_size-1.0, 0.01)    
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[0].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[0].plot_surface(x,y,z, color='white', alpha=0.01)

            #Quadratic subplot
            ax[1].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[1].set_xticks(np.arange(0, world_size+1, 1))
            ax[1].set_xticklabels('', fontsize=10)
            ax[1].set_yticklabels('', fontsize=10)
            ax[1].set_yticks(np.arange(0, world_size+1, 1))
            ax[1].set_zlim(-1.0,1.0)
            ax[1].set_zticklabels([''], fontsize=10)
            ax[1].view_init(elev=30, azim=-65)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))
            grid = np.arange(0.0, world_size-1.0, 0.01)           
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[1].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[1].plot_surface(x,y,z, color='white', alpha=0.01)

            #Quadratic subplot
            ax[2].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[2].set_xticks(np.arange(0, world_size+1, 1))
            ax[2].set_xticklabels('', fontsize=10)
            ax[2].set_yticklabels('', fontsize=10)
            ax[2].set_yticks(np.arange(0, world_size+1, 1))
            ax[2].set_zlim(-1.0,1.0)
            ax[2].set_zticklabels([''], fontsize=10)
            ax[2].view_init(elev=30, azim=-45)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))
            grid = np.arange(0.0, world_size-1.0, 0.01)           
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[2].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[2].plot_surface(x,y,z, color='white', alpha=0.01)

            #Quadratic subplot
            ax[3].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[3].set_xticks(np.arange(0, world_size+1, 1))
            ax[3].set_xticklabels('', fontsize=10)
            ax[3].set_yticklabels('', fontsize=10)
            ax[3].set_yticks(np.arange(0, world_size+1, 1))
            ax[3].set_zlim(-1.0,1.0)
            ax[3].set_zticklabels([''], fontsize=10)
            ax[3].view_init(elev=30, azim=-25)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))
            grid = np.arange(0.0, world_size-1.0, 0.01)           
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[3].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[3].plot_surface(x,y,z, color='white', alpha=0.01)

            #Save the figure
            fig.tight_layout()
            fig.savefig(filename, dpi=300) #, bbox_inches='tight')


def main():
    env=init_env()
    my_mlp=MLP(tot_inputs=2,tot_hidden=2,tot_outputs=1,activation="tanh")
    learning_rate = 0.1
    gamma = 0.9
    tot_epoch = 10001
    print_epoch = 100

    for epoch in range(tot_epoch):
       
        observation=env.reset(exploring_starts=True)

        for step in range(1000):
            action=np.random.randint(0,4)
            new_observation,reward,done=env.step(action)
            my_mlp,error=update(my_mlp, new_observation,reward,learning_rate,gamma,done)
    
            if done: break
        if(epoch % print_epoch == 0 and epoch!=0):
            print("")
            print("Epoch: " + str(epoch+1))
            print("Tot steps: " + str(step))
            print("Error: " + str(error))

            print_utility(my_mlp, tot_rows=5, tot_cols=5)
    print("Generating plot, please wait...")
    subplot(my_mlp, world_size=5, filename="xor_planes.png")
    print("Done!")

if __name__ == "__main__":
    main()