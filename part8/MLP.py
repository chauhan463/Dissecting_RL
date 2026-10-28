import numpy as np

class MLP():
    def __init__(self,tot_inputs,tot_hidden,tot_outputs,activation='sigmoid'):
        if (activation!='sigmoid' and activation!='tanh'):
            raise ValueError("The activation function doesnt exist"+str(activation))
        else:
            self.activation=activation

        self.tot_inputs=tot_inputs
        self.W1 = np.random.normal(0.0,0.1,(tot_inputs+1,tot_hidden))
        self.W2= np.random.normal(0.0,0.1,(tot_hidden+1,tot_outputs))
        self.tot_outputs=tot_outputs

    def _sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def _tanh(self,z):
        return np.tanh(z)

    def forward(self,x,verbose=False):
        if(x.shape[0]!=self.tot_inputs):raise ValueError("The size of X is wrong")
        self.x=np.hstack([x,np.array([1.0])])
        self.z1=np.dot(self.x,self.W1)
        if self.activation=="sigmoid":
            self.h=self._sigmoid(self.z1)
        else:
            self.h=self._tanh(self.z1)

        self.h=np.hstack([self.h,np.array([1.0])])
        self.z2=np.dot(self.h,self.W2)
        if self.activation=="sigmoid":
            self.y=self._sigmoid(self.z2)
        else:
            self.y=self._tanh(self.z2)
        if(verbose): print("z1: " + str(self.z1))
        if(verbose): print("h: " + str(self.h))
        if(verbose): print("z2: " + str(self.z2))
        if(verbose): print("y: " + str(self.y))
        if(verbose): print("W1: " + str(self.W1))
        if(verbose): print("W2: " + str(self.W2))
        return self.y



    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1.0 - self._sigmoid(z))

    def _tanh_derivative(self, z):
        return 1 - np.square(np.tanh(z))

    def backward(self,y,target,verbose=False):
        if(y.shape!=target.shape):raise ValueError("[ERROR] The size of target is wrong!")
        dE_dY=-(target-y)
        if self.activation=="sigmoid":
            dY_dz2=self._sigmoid_derivative(self.z2)
        else:
            dY_dz2=self._tanh_derivative(self.z2)
        dz2_dW2=self.h
        dz2_dh=self.W2
        if(self.activation=="sigmoid"):
            dh_dz1 = self._sigmoid_derivative(self.z1) 
        elif(self.activation=="tanh"):
            dh_dz1 = self._tanh_derivative(self.z1)   
        dz1_dW1 = self.x

        dE_dW2=np.dot(np.expand_dims(dE_dY*dY_dz2,axis=1),np.expand_dims(dz2_dW2,axis=0)).T

        dE_dW1=(dE_dY * dY_dz2) 
        dE_dW1=np.dot(dE_dW1,dz2_dh.T)[0:-1]*dh_dz1
        dE_dW1=np.dot(np.expand_dims(dE_dW1,axis=1),np.expand_dims(dz1_dW1,axis=0)).T
        if(verbose): print("dE_dW1: " + str(dE_dW1))
        if(verbose): print("dE_dW2: " + str(dE_dW2))
        return dE_dW1, dE_dW2

    def train(self,x,target,learning_rate=0.1):
        y=self.forward(x)
        dE_dW1, dE_dW2 = self.backward(y, target)

        self.W2=self.W2-learning_rate*dE_dW2
        self.W1=self.W1-learning_rate*dE_dW1

        error=0.5*(target-y)**2
        return error

def main():
   my_mlp = MLP(tot_inputs=6, tot_hidden=64, tot_outputs=3, activation="tanh")  
   x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
   target = np.array([-0.75, 0.25, 0.10])
   for i in range(500):
      y = my_mlp.forward(x, verbose=False)
      print("[step " + str(i) + "] y=" + str(my_mlp.y))
      my_mlp.backward(y, target, verbose=False)
      my_mlp.train(x, target, learning_rate=0.1)
      print
       
       

if __name__=="__main__":
    main()
                