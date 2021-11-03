# Mcculloch pitts ANN written by Ali Seyedalian.
# 810199194

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1 

x1 , x2 = input("Enter x1 x2 : ").split() # -2 3
x1 = float(x1)
x2 = float(x2)

h1 = sign(5*x1-x2+3)
h2 = sign(-x2+3)
h3 = sign(-5*x1-2*x2+21)
h4 = sign(x2+2)
#print(h1,h2,h3,h4)

f = sign(h1+h2+h3+h4-3)
if f == 1:
    print("+1 : inside")
else:
    print("-1 : outside")