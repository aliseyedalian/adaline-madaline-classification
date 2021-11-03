import numpy as np
import matplotlib.pyplot as plt



def init_train_set():
    train_set = list()
    n1=100
    x , y = np.random.normal(loc=2,scale=0.5,size=n1) , np.random.normal(loc=0,scale=0.2,size=n1)
    plt.scatter(x,y)
    for i in range(n1):
        train_set.append({"x1" : x[i],"x2" :y[i], "t" : +1})
    n2=30
    x , y = np.random.normal(loc=0,scale=0.1,size=n2) , np.random.normal(loc=1,scale=0.7,size=n2)
    plt.scatter(x,y)
    for i in range(n2):
        train_set.append({"x1" : x[i],"x2" :y[i], "t" : -1})
    return train_set

def plot_results(epoch,alpha):
    plt.title("Adaline (Sign)\nepochs="+str(epoch)+", alpha="+str(alpha))
    x = np.linspace(-1,4)
    label = 'x2 = '+str(round(-w1/w2,3))+' * x1 + '+str(round(-b/w2,3))+'\nw1 = '+str(round(w1,3))+\
    ' ,w2 = '+str(round(w2,3))+' ,b = '+str(round(b,3))
    plt.text(1, 3, label, fontsize = 9)
    plt.xlim(-1, 4)
    plt.ylim(-1,4)
    plt.axhline(y=0, color='gray')
    plt.axvline(x=0, color='gray')
    plt.plot(x,(-w1/w2)*x-b/w2,'-r')
    plt.xlabel("x1")
    plt.ylabel("x2",rotation=0)
    plt.show()
    plt.title("Adaline (Sign)\nepochs="+str(epoch)+", alpha="+str(alpha))
    plt.plot(sum_errors_epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Function")
    plt.show()


def f(net): # Activation Function sign
    return 1 if net>=0 else -1

def allow_start_next_epoch():
    response = False
    sum_errors = 0
    for p in train_set:
        x1 , x2 , t = p["x1"] , p["x2"] , p["t"]
        net = w1*x1 + w2*x2 + b
        h = f(net)
        error_p = 0.5*(t - h)**2
        if error_p != 0:
            response = True
        sum_errors += error_p 
    sum_errors_epochs.append(sum_errors)
    return response

if __name__ == '__main__':
    np.random.seed(seed=3)
    w1 = np.random.rand()
    w2 = np.random.rand()
    b = np.random.rand()
    alpha = 0.01
    sum_errors_epochs = []
    train_set = init_train_set()
    epoch = 0
    while allow_start_next_epoch():
        epoch+=1
        for p in train_set:
            x1 , x2 , t = p["x1"] , p["x2"] , p["t"] 
            net = w1*x1 + w2*x2 + b
            h = f(net)
            w1 = w1 + alpha*(t-h)* x1
            w2 = w2 + alpha*(t-h)* x2
            b  = b  + alpha*(t-h)

    plot_results(epoch,alpha)
    
        