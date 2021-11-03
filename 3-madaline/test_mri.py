import numpy as np
import matplotlib.pyplot as plt
import math


def f(net):
    gama = 200
    return math.tanh(gama*net)
    
def calc_net(b,w,x):
    return b + np.dot(np.transpose(x),w)

class Adaline:
    def __init__(self,w,b,alpha=0.5,name=""):
        self.w = w # np.array
        self.b = b
        self.alpha = alpha
        self.net = None
        self.h = None
        self.name = name
    def updateWeights(self,x,t):
        self.net = calc_net(b=self.b , w=self.w , x=x)
        self.h = f(self.net)
        self.b = self.b + self.alpha * (t - self.h)
        self.w = self.w + self.alpha * (t - self.h) * x

def TEST_MRI():
    # STEP 0 
    train_set_xor = [{"x1" :0.5,"x2" :0.5, "t" : -1},{"x1" :-0.5,"x2" :-0.5, "t" : -1},{"x1" :0,"x2" :0, "t" : -1},{"x1" :-0.5,"x2" :1, "t" : -1},
    {"x1" :1,"x2" :-1, "t" : 1},{"x1" :-1,"x2" :1, "t" : 1},{"x1" :2,"x2" :2, "t" : 1},{"x1" :-2,"x2" :-2, "t" : 1},{"x1" :0,"x2" :-2, "t" : 1},
    {"x1" :0,"x2" :3, "t" : 1},{"x1" :-3,"x2" :0, "t" : 1}]
    plt.scatter([0,0.5,-0.5,-0.5],[0,0.5,-0.5,1],c="red",s=6) # t = -1
    plt.scatter([1,-1,2,-2,0,0,-3],[-1,1,2,-2,-2,3,0],c="blue",s=6) # t = 1
    Z1 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=0.1)
    Z2 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=0.1)
    Z3 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=0.1)
    learning_neurons = [Z1,Z2,Z3]
    Y = Adaline(w=np.array([0.5,0.5,0.5]),b=1)
    # STEP 1
    next_epoch = True
    error_rates = []
    while next_epoch:
        next_epoch = False
        error_rate = 0
        # STEP 2
        for p in train_set_xor:
            # STEP 3
            x, t = np.array([p["x1"] , p["x2"]]), p["t"]
            # STEP 4
            Z1.net = calc_net(Z1.b,Z1.w,x)
            Z2.net = calc_net(Z2.b,Z2.w,x)
            Z3.net = calc_net(Z3.b,Z3.w,x)
    
            # STEP 5
            Z1.h = f(Z1.net)
            Z2.h = f(Z2.net)
            Z3.h = f(Z3.net)
        
            # STEP 6
            Y.net = calc_net(Y.b,Y.w,x=np.array([Z1.h,Z2.h,Z3.h]))
            Y.h = f(Y.net)
            # STEP 7
            if t != Y.h:
                next_epoch = True
                error_rate+=1
                if t == 1:
                    Z = min(learning_neurons, key=lambda neuron: (0-neuron.net)**2 )
                    Z.updateWeights(x,t)      
                elif t == -1:
                    for Z in learning_neurons:
                        if Z.net > 0: Z.updateWeights(x,t)
                        
        error_rates.append(error_rate)
    x = np.linspace(-10,10)
    plt.plot(x,(-Z1.w[0]/Z1.w[1])*x-Z1.b/Z1.w[1],'-g')
    plt.plot(x,(-Z2.w[0]/Z2.w[1])*x-Z2.b/Z2.w[1],'-g')
    plt.plot(x,(-Z3.w[0]/Z3.w[1])*x-Z3.b/Z3.w[1],'-g')

    plt.axhline(y=0, color='gray')
    plt.axvline(x=0, color='gray')
    plt.xlabel("x1")
    plt.ylabel("x2",rotation=0)
    plt.xlim(-4, 4)
    plt.ylim(-4,4)
    plt.show()
    plt.plot(error_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Function")
    plt.show()



if __name__ == '__main__':
    TEST_MRI()
    



    
    
    
    
    