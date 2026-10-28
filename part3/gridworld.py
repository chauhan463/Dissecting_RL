import numpy as np

class Gridworld:
    def __init__(self,tot_row, tot_col):
        self.actions_size= 4
        self.world_rows= tot_row
        self.world_cols= tot_col
        self.transition_matrix= np.ones((self.actions_size,self.actions_size))/self.actions_size

        self.reward_matrix=np.zeros((self.world_rows,self.world_cols))
        self.state_matrix=np.zeros((self.world_rows,self.world_cols))
        self.position=[np.random.randint(0,self.world_rows), np.random.randint(0,self.world_cols)]
      
    def setTransitionMatrix(self,transition_matrix):
        if(transition_matrix.shape != self.transition_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.transition_matrix = transition_matrix

    def setRewardMatrix(self,reward_matrix):
        if(reward_matrix.shape != self.reward_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.reward_matrix = reward_matrix

    def setStateMatrix(self,state_matrix):
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.state_matrix = state_matrix

    def setPosition(self, index_row=None, index_col=None):
        if(index_row is None or index_col is None): self.position = [np.random.randint(self.world_rows), np.random.randint(self.world_cols)]
        else: self.position = [index_row, index_col]

    def reset(self, exploring_starts=False):
        if exploring_starts:
            while True:
                row = np.random.randint(0, self.world_rows)
                col = np.random.randint(0, self.world_cols)
                if self.state_matrix[row,col]==0:
                    break
            self.position=[row,col]
        else:
            self.position=[self.world_rows-1,0]
        return self.position

    def render(self):
        graph = ""
        for row in range(self.world_rows):
            row_string = ""
            for col in range(self.world_cols):
                if(self.position == [row, col]): row_string += u" \u25CB " # u" \u25CC "
                else:
                    if(self.state_matrix[row, col] == 0): row_string += ' - '
                    elif(self.state_matrix[row, col] == -1): row_string += ' # '
                    elif(self.state_matrix[row, col] == +1): row_string += ' * '
            row_string += '\n'
            graph += row_string 
        print (graph)            

    
    def step(self, action):

        if action >=self.actions_size:
            raise ValueError("Choosing invalid action")

        actual_action = np.random.choice(4,1,p=self.transition_matrix[int(action),:])
        curr_position=self.position
        if actual_action == 0: new_position = [curr_position[0]-1,curr_position[1]]
        elif actual_action== 1: new_position = [curr_position[0],curr_position[1]+1]
        elif actual_action== 2: new_position = [curr_position[0]+1,curr_position[1]]
        elif actual_action== 3: new_position = [curr_position[0],curr_position[1]-1]
        else:raise ValueError("Choosing invalid action")

        x,y=new_position

        if (x>=0 and x<self.world_rows) and (y>=0 and y<self.world_cols) and (self.state_matrix[x,y]!=-1):
            self.position=[x,y]

        reward=self.reward_matrix[self.position[0],self.position[1]]
        done = bool(self.state_matrix[self.position[0],self.position[1]])
            
        return self.position, reward, done



