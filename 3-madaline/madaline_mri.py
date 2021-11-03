import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from math import tanh

def sample_ring(center,r1,r2,n_points):
    nd = center.size
    x = np.random.normal(size=(n_points,nd),scale=0.5)
    sq = np.sum(x**2,axis=1)
    z = (r2-r1)/r2
    fr = (r2-r1)*gammainc(nd/2**z,sq/2**z)**(1/nd)/np.sqrt(sq) + r1/np.sqrt(sq)
    frtiled = np.tile(fr.reshape(n_points,1),(1,nd))
    p = center + np.multiply(x,frtiled)
    return p

def init_train_sets(plot=True): 
    train_set1 = [] # contains all data (red group & blue group & ring group)
    train_set2 = [] # contains inner data (red group & blue group)
    n1=100 # Right blue group
    x_blues , y_blues = np.random.normal(loc=3,scale=0.5,size=n1) , np.random.normal(loc=0,scale=0.5,size=n1)
    for i in range(n1): # Right blue group
        x0 , x1 = x_blues[i] , y_blues[i] 
        train_set1.append({"x0":x0 , "x1":x1 , "t":-1 })
        train_set2.append({"x0":x0 , "x1":x1 , "t":-1 })
    n2=100 # Left red group
    x_reds , y_reds = np.random.normal(loc=0,scale=0.5,size=n2) , np.random.normal(loc=0,scale=0.5,size=n2)
    for i in range(n2): # Left red group
        x0 , x1 = x_reds[i] , y_reds[i]
        train_set1.append({"x0":x0 , "x1":x1 , "t":-1 })
        train_set2.append({"x0":x0 , "x1":x1 , "t":+1 })
    n3=250 # ring group
    p = sample_ring(center=np.array([1.5,0]),r1=4,r2=6,n_points=n3)
    for i in range(n3):
        x0 , x1 = p[i][0] , p[i][1]
        train_set1.append({"x0":x0 , "x1":x1 , "t":+1 })
    if plot == True:
        plt.scatter(x_blues,y_blues,c="blue",s=6)
        plt.scatter(x_reds,y_reds,c="red",s=6)
        fig = plt.figure(1)
        ax1 = fig.gca()
        ax1.scatter(p[:,0],p[:,1],s=6,c="green")
        ax1.set_aspect('equal')
    return train_set1 , train_set2

def f(net):
    return 1 if net>=0 else -1
    
def calc_net(b,w,x):
    return b + np.dot(np.transpose(x),w)

class Adaline:
    def __init__(self,w,b,alpha=0.05):
        self.w = w # weights, np.array
        self.b = b # bias
        self.alpha = alpha # learning rate
        self.net = 0
        self.h = 0  
    def updateWeights(self,x,t): 
        self.net = calc_net(b=self.b , w=self.w , x=x)
        self.h = f(self.net)
        self.b = self.b + self.alpha * (t - self.h)
        self.w = self.w + self.alpha * (t - self.h) * x

def plot_results(Z_set,total_epochs,alpha):
    x = np.linspace(-10,10)
    plt.xlim(-5, 8)
    plt.ylim(-7, 6)
    t='Madaline Neural Network Classification\nalpha='+str(alpha)+'  epochs='+str(total_epochs) 
    plt.title(t)
    for Z in Z_set:
        plt.plot(x,(-Z.w[0]/Z.w[1])*x - Z.b/Z.w[1])
    plt.show()

def main(alpha = 0.02,plot = True,seed = 0):
    if seed:np.random.seed(seed)
    train_set1 , train_set2 = init_train_sets(plot = plot)
    total_epochs = 0
    '''-- Madaline NN (MRI algorithm) for separating ring group from other by 3 line --'''  
    Z0 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=alpha)
    Z1 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=alpha)
    Z2 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=alpha)
    learning_neurons = [Z0,Z1,Z2]
    Y0 = Adaline(w=np.array([0.5,0.5,0.5]),b=1) 
    next_epoch = True
    while next_epoch:
        total_epochs += 1 
        next_epoch = False
        for p in train_set1:
            x, t = np.array([p["x0"] , p["x1"]]), p["t"]
            Z0.net , Z1.net , Z2.net = calc_net(Z0.b,Z0.w,x) , calc_net(Z1.b,Z1.w,x) , calc_net(Z2.b,Z2.w,x)
            Z0.h , Z1.h , Z2.h = f(Z0.net) , f(Z1.net) , f(Z2.net)   
            Y0.net = calc_net(Y0.b,Y0.w,x=np.array([Z0.h,Z1.h,Z2.h]))
            Y0.h = f(Y0.net)
            if t != Y0.h:
                next_epoch = True
                if t == 1:
                    Z = min(learning_neurons, key=lambda neuron: (0-neuron.net)**2 )
                    Z.updateWeights(x,t)      
                elif t == -1:
                    for Z in learning_neurons:
                        if Z.net > 0: Z.updateWeights(x,t)   

    ''' -- A Simple Adaline Neuron for Separating Red Group from Blue Group by 1 line -- '''
    Z3 = Adaline(w=np.random.rand(2),b=np.random.rand(),alpha=alpha)
    next_epoch = True
    while next_epoch:
        total_epochs += 1 
        next_epoch = False
        for p in train_set2:
            x, t = np.array([p["x0"] , p["x1"]]), p["t"] 
            Z3.updateWeights(x,t)
        for p in train_set2:
            x, t = np.array([p["x0"] , p["x1"]]), p["t"] 
            h = f(calc_net(Z3.b,Z3.w,x))
            error_p = 0.5*(t - h)**2
            if error_p != 0:
                next_epoch = True
                break  

    if plot == True: 
        plot_results([Z0,Z1,Z2,Z3],total_epochs,alpha)
    return total_epochs 
    
       
if __name__ == '__main__':
    main(alpha=0.01,plot=True,seed = 6)
    exit()
    alphas = [0.005,0.006,0.007,0.008,0.009,0.01,0.05,0.1,0.5]
    epochs_list = []
    for a in alphas:
        epochs = main(alpha = a , plot = False,seed =4)
        epochs_list.append(epochs)

    #sum(epochs_list)
    props = [item/1000 for item in epochs_list]
    plt.xlabel("learning rate (alpha)")
    plt.ylabel("#epochs / 1000")
    plt.xticks(alphas)
    plt.plot(alphas,props)
    plt.show()