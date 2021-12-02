import numpy as np
import matplotlib.pyplot as plt

L = 200
W = 100
r = 25
q = 2.3

x1 = L//2 - L//r
x2 = L//2 + L//r
y1 = W//q
y2 = W - W//q

def gate_bottom(pos) :
    x,y = pos
    a1 = y1*(L**2/4+(x1**2-x2**2)/(x2-x1)*L/2-x1**2-(x1**2-x2**2)/(x2-x1)*x1)**(-1)
    b1 = a1*(x1**2-x2**2)/(x2-x1)
    c1 = -(a1*x1**2+b1*x1)
    ymax = a1*x**2+b1*x+c1  
    return y <= ymax
def gate_top(pos) :
    x,y = pos    
    a2 = -y1*(L**2/4+(x1**2-x2**2)/(x2-x1)*L/2-x1**2-(x1**2-x2**2)/(x2-x1)*x1)**(-1)
    b2 = a2*(x1**2-x2**2)/(x2-x1)
    c2 = W-a2*x1**2-b2*x1
    ymax = a2*x**2+b2*x+c2 
    return y >= ymax
def cond(pos) :
	return gate_bottom(pos) or gate_top(pos)
for i in range(W) : 
    for j in range(L) : 
        if gate_bottom((j,i)) or gate_top((j,i)) :
            plt.plot(j,i,'or')
plt.show()