# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:45:30 2021

@author: adrie
"""
import kwant
import scipy
import numpy as np
from matplotlib import pyplot as plt

##### 2.1 #####

a = 1 # a is the lattice parameter
t = 1 # t is the hopping parameter

lat = kwant.lattice.square(a)

def make_lead_x(W):
    syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
    syst[(lat(0, y) for y in range(W))] = 4 * t
    syst[lat.neighbors()] = -t
    return syst

def scattering(W, L):
    # Construct the scattering region.
    sr = kwant.Builder()
    sr[(lat(x, y) for x in range(L) for y in range(W))] = 4*t
    sr[lat.neighbors()] = -t   
    # Build and attach lead from both sides.
    lead = make_lead_x(W)
    sr.attach_lead(lead)
    sr.attach_lead(lead.reversed())
    return sr

# make_QPC1 takes as input a lattice (square), its dimensions (W,L), 2 parameters defining the size 
# of the QPC (r,q) and the value to give to the lattice nodes symbolising the QPC (QPC_pot).
# The dimensions of the QPC are (2*L/r x 2*W/q) centered at x=L/2 and for 0<y<W/q and W(q+2)/2<y<W.
# It returns the indices needed by make_QPC2 to build the smooth transition zone.

def make_QPC1(square, W, L, r, q, QPC_pot):
    x1 = L//2 - L//r
    x2 = L//2 + L//r
    y1 = W//q
    y2 = W - W//q
    for i in range(W):
        for j in range(L):
            if x1<j<x2 and (i<y1 or i>=y2):
                square[lat(j,i)] = QPC_pot
    return x1, x2, y1, y2

# make_QPC2 is responsible for the decrease of on-site parameter in order to have a smoother 
# transition between the gates and the channel. This is much more realistic than having an abrupt 
# transition. It takes as input a lattice (square), its dimensions (W,L), the indices of the lines 
# where to change the on-site and the new value of on-site (pot) to apply.
# It returns the indices of the next lines to be updated.
# This function should be applied several times in a row in order to produce the smooth transition.

def make_QPC2(square, W, L, x1, x2, y1, y2, pot):
    y2 = y2-1
    for i in range(W):
        for j in range(L):
            if (j==x1 or j==x2) and (i<y1 or i>y2):
                square[lat(j,i)] = pot
            if (i==y1 or i==y2) and (x1<=j<=x2):
                square[lat(j,i)] = pot
    return x1-1, x2+1, y1+1, y2

def make_QPC_lin(square, W, L, r, q, QPC_pot, iterations, t): # Linear decrease
    x1, x2, y1, y2 = make_QPC1(square, W, L, r, q, QPC_pot)
    n = iterations-1
    m = (4*t-QPC_pot)/(n)
    for i in range(1, iterations):
        next_x1, next_x2, next_y1, next_y2 = make_QPC2(square, W, L, x1, x2, y1, y2, QPC_pot+m*i)
        x1, x2, y1, y2 = next_x1, next_x2, next_y1, next_y2

def make_QPC_quad(square, W, L, r, q, QPC_pot, iterations, t): # Quadratic decrease
    x1, x2, y1, y2 = make_QPC1(square, W, L, r, q, QPC_pot)
    n = iterations-1
    b = (4*n*t + 2*n*t*(QPC_pot/t)**0.5)/(QPC_pot - 4*t)
    a = b**2*QPC_pot
    x = np.arange(0, iterations, 1)
    quad_dec = a/(x+b)**2
    for i in range(1, iterations):
        next_x1, next_x2, next_y1, next_y2 = make_QPC2(square, W, L, x1, x2, y1, y2, quad_dec[i])
        x1, x2, y1, y2 = next_x1, next_x2, next_y1, next_y2

W = 20
L = 50
square = scattering(W,L)
square2 = scattering(W,L)
# x1, x2, y1, y2 = make_QPC1(square2, W, L, 6, 4, 10)
# next_x1, next_x2, next_y1, next_y2 = make_QPC2(square2, W, L, x1, x2, y1, y2, 8)
# make_QPC2(square2, W, L, next_x1, next_x2, next_y1, next_y2, 6)
make_QPC_quad(square2, W, L, 25, 6, 25, 8, 1)
mat = np.zeros((W,L))
for i in range(L):
    for j in range(W):
        # print("i,j = {},{}".format(i,j))
        mat[j][i] = square2[lat(i,j)]

##### Plot of the on-site parameter in the lattice #####

fig, ax = plt.subplots()

x = np.arange(0, L, 1)
y = np.arange(0, W, 1)
xx, yy = np.meshgrid(x, y)
ax.scatter(xx, yy)
plt.contourf(x, y, mat)
plt.show()

##### Parabolic gates #####

x0 = 2
r = 25
q = 15
n = 3
# X = np.arange(x0, x0+2*(n+1), 1)
# dY = np.zeros(2*n+1)
# for i in range(n):
#     dY[i] = 2**(-i+1)
#     dY[-(i+1)] = -dY[i]
# Y = np.zeros(len(dY)+1)
# for i in range(1,len(Y)):
#     Y[i] = Y[i-1] + dY[i-1]
# plt.plot(X,Y)
# plt.grid()

def make_parabola(x0, n, e, W):
    x = np.arange(x0, x0+2*(n+1), 1)
    dy = dY = np.zeros(2*n+1)
    y = np.zeros(len(dY)+1)
    Y = np.zeros(len(dY)+1)
    for i in range(n):
        dy[i] = e*2**(-i)
        dy[-(i+1)] = -dy[i]
    for i in range(len(y)):
        if i==0:
            y[i] = 0
            Y[i] = W
        else:
            y[i] = y[i-1] + dy[i-1]
            Y[i] = W - y[i]
    plt.plot(x, y)
    plt.plot(x, Y)
    plt.grid()
    return x, y, dy, Y

# X, Y, dY, Yy = make_parabola(x0, n, n+1, W)
# plt.figure()
# plt.plot(X, Y)
# plt.plot(X, Yy)
# plt.grid()
m=3
# for i in range(m):
#     make_parabola(x0+m-i, i, i+1, W)

# kwant.plot(square2)
sys = square2.finalized()
E = 0.8
smatrix = kwant.smatrix(sys, energy = E)
T = smatrix.transmission(1, 0)
# print('T = '+str(T))

# The function T_of_E is quite heavy and takes some time to execute so I should be careful when 
# calling it with large inputs
# kwant.smatrix().transmission() gives the conductance of the system (see the tutorial section 2.2)

def T_of_E(E_m, E_M, sys, title): # Plots the relation T(E)
    Energy = np.linspace(E_m, E_M, 100)
    Trans = np.zeros(len(Energy))
    for i in range(len(Energy)):
        Trans[i] = kwant.smatrix(sys, energy = Energy[i]).transmission(1,0)
        print(str(i)+'/'+str(len(Energy)))
    plt.figure()
    plt.plot(Energy, Trans)
    plt.grid()
    plt.xlabel("Energy [t]")
    plt.ylabel("Conductance [e²/h]")
    plt.title(title)

T_of_E(0, 5, sys, "W = "+str(W)+" ; L = "+str(L))

##### 2.3 #####

# This code comes straight from the on-line kwant tutorial

def make_lead(a=1, t=1.0, W=10): # Default values are a=1, t=1, W=10
    # Start with an empty lead with a single square lattice
    lat = kwant.lattice.square(a)

    sym_lead = kwant.TranslationalSymmetry((-a, 0))
    lead = kwant.Builder(sym_lead)

    # build up one unit cell of the lead, and add the hoppings
    # to the next unit cell
    for j in range(W):
        lead[lat(0, j)] = 4 * t

        if j > 0:
            lead[lat(0, j), lat(0, j - 1)] = -t

        lead[lat(1, j), lat(0, j)] = -t

    return lead

# Remark : the Band structure plotted below is the one of the leads and is not influenced by the 
# scattering region (I think). That's why the conductance plt changes even though the Band structure 
# does not => because the band structure and the lead are no longer identical. I can't say that the 
# conductance increases by 1 quanta at each energy level crossed because I don't have the corresponding 
# band structure anymore.
def main():
    lead = make_lead().finalized()
    kwant.plotter.bands(lead, show=True)
    plt.grid()
    plt.xlabel("momentum [(lattice constant)^-1]")
    plt.ylabel("energy [t]")
    plt.title("Band structure")
    plt.show()

# main()

# kwant.plotter.bands needs a builder.InfiniteSystem object as input, sys is not one of those.
# kwant.plotter.bands(sys, show=True)

##### 2.4 #####

T_of_E()