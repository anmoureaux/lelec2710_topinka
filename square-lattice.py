# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:45:30 2021

@author: adrie
"""
import kwant
import scipy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

##### 2.1 #####

a = 1 # a is the lattice parameter
t = 1 # t is the hopping parameter

lat = kwant.lattice.square(a, norbs=1) # to decomment if problem

def make_lead_x(W):
    syst = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
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

############### Rectangular gates ################################

# make_QPC1 takes as input a lattice (square), its dimensions (W,L), 2 parameters defining the size 
# of the QPC (r,q) and the value to give to the lattice nodes symbolising the QPC (QPC_pot).
# The dimensions of the QPC are (2*L/r x 2*W/q) centered at x=L/2 and for 0<y<W/q and W-W/q<y<W.
# It returns the indices needed by make_QPC2 to build the smooth transition zone.

def make_QPC1_rect(square, W, L, r, q, QPC_pot):
    x1 = L//2 - L//r
    x2 = L//2 + L//r
    y1 = W//q
    y2 = W - W//q
    for i in range(W):
        for j in range(L):
            if x1<j<x2 and (i<y1 or i>=y2):
                square[lat(j,i)] = QPC_pot
    return x1,x2,y1,y2

# make_QPC2 is responsible for the decrease of on-site parameter in order to have a smoother 
# transition between the gates and the channel. This is much more realistic than having an abrupt 
# transition. It takes as input a lattice (square), its dimensions (W,L), the indices of the lines 
# where to change the on-site and the new value of on-site (pot) to apply.
# It returns the indices of the next lines to be updated.
# This function should be applied several times in a row in order to produce the smooth transition.

def make_QPC2_rect(square, W, L, x1, x2, y1, y2, pot):
    for i in range(W):
        for j in range(L):
            if (j==x1 or j==x2) and (i<=y1 or i>=y2):
                square[lat(j,i)] = pot
            if (i==y1 or i==y2) and (x1<=j<=x2):
                square[lat(j,i)] = pot
    return x1-1, x2+1, y1+1, y2-1

def make_QPC_rect_lin(square, W, L, r, q, QPC_pot, iterations, t): # Linear decrease
    x1,x2,y1,y2 = make_QPC1_rect(square, W, L, r, q, QPC_pot)
    m = (QPC_pot-4*t)/(iterations)
    for i in range(1, iterations):
        x1, x2, y1, y2 = make_QPC2_rect(square, W, L, x1, x2, y1, y2, QPC_pot-m*i)
         
# The 'iterations' parameter defines the length between the source at QPC_pot and the channel.
# QPC_pot is the highest value of on-site (the one on the gates)
def make_QPC_rect_quad(square, W, L, r, q, QPC_pot, iterations, t): # Quadratic decrease
    x1, x2, y1, y2 = make_QPC1_rect(square, W, L, r, q, QPC_pot)
    n = iterations-1
    b = (4*n*t + 2*n*t*(QPC_pot/t)**0.5)/(QPC_pot - 4*t)
    a = b**2*QPC_pot
    x = np.arange(0, iterations, 1)
    quad_dec = a/(x+b)**2 # quad_dec is the quadratically decreasing potential
    for i in range(1, iterations):
        x1, x2, y1, y2 = make_QPC2_rect(square, W, L, x1, x2, y1, y2, quad_dec[i])

################## Parabolic gates #################################""

def make_QPC1_parabola(square, W, L, r, q, QPC_pot):
    x1 = L//2 - L//r
    x2 = L//2 + L//r
    y1 = W//q
    y2 = W - W//q
    #bottom gate parabola
    a1 = y1*(L**2/4+(x1**2-x2**2)/(x2-x1)*L/2-x1**2-(x1**2-x2**2)/(x2-x1)*x1)**(-1)
    b1 = a1*(x1**2-x2**2)/(x2-x1)
    c1 = -(a1*x1**2+b1*x1)
    #top gate parabola
    a2 = -y1*(L**2/4+(x1**2-x2**2)/(x2-x1)*L/2-x1**2-(x1**2-x2**2)/(x2-x1)*x1)**(-1)
    b2 = a2*(x1**2-x2**2)/(x2-x1)
    c2 = W-a2*x1**2-b2*x1
    for i in range(W) : 
        for j in range(L) : 
            if i <= a1*j**2+b1*j+c1 or i >= a2*j**2+b2*j+c2 :
                square[lat(j,i)] = QPC_pot
    return x1,y1,x2,y2

def make_QPC2_parabola(square, W, L, x1, y1, x2, y2, pot, tol) :
    #bottom gate parabola
    a1 = y1*(L**2/4+(x1**2-x2**2)/(x2-x1)*L/2-x1**2-(x1**2-x2**2)/(x2-x1)*x1)**(-1)
    b1 = a1*(x1**2-x2**2)/(x2-x1)
    c1 = -(a1*x1**2+b1*x1)
    #top gate parabola
    a2 = -y1*(L**2/4+(x1**2-x2**2)/(x2-x1)*L/2-x1**2-(x1**2-x2**2)/(x2-x1)*x1)**(-1)
    b2 = a2*(x1**2-x2**2)/(x2-x1)
    c2 = W-a2*x1**2-b2*x1
    for i in range(W):
        for j in range(L):
            if abs(a1*j**2+b1*j+c1 - i)<=tol or abs(a2*j**2+b2*j+c2 - i)<=tol : 
                square[lat(j,i)] = pot
    x1 -= 1
    x2 += 1
    y1 += 0.7
    y2 -= 0.7
    return x1,y1,x2,y2

def make_QPC_parabola_lin(square, W, L, r, q, QPC_pot, iterations, t): # Linear decrease
    x1,y1,x2,y2 = make_QPC1_parabola(square, W, L, r, q, QPC_pot)
    tol = 4
    m = (QPC_pot-4*t)/(iterations)
    for i in range(1, iterations):
        x1,y1,x2,y2 = make_QPC2_parabola(square, W, L,x1,y1,x2,y2,QPC_pot-m*i,tol)

def make_QPC_parabola_quad(square, W, L, r, q, QPC_pot, iterations, t): # Quadratic decrease
    x1, x2, y1, y2 = make_QPC1_parabola(square, W, L, r, q, QPC_pot)
    tol = 4
    n = iterations
    b = (4*n*t + 2*n*t*(QPC_pot/t)**0.5)/(QPC_pot - 4*t)
    a = b**2*QPC_pot
    x = np.arange(0, iterations, 1)
    quad_dec = a/(x+b)**2 # quad_dec is the quadratically decreasing potential
    for i in range(1, iterations):
        x1, x2, y1, y2 = make_QPC2_parabola(square, W, L, x1, x2, y1, y2, quad_dec[i], tol)

##### Modelisation of the SPM tip #####

# The 3 functions below work according to the same structure as the ones for the gates (make_QPC_x).

# The 'onsite' parameter is equivalent to 'QPC_pot' and 'pot', they all represent the same thing.
def make_circle1(square, x0, y0, R, onsite):
    xm = int(np.floor(x0-R))
    xM = int(np.ceil(x0+R))
    ym = int(np.floor(y0-R))
    yM = int(np.ceil(y0+R))
    for i in range(xm, xM+1): # i is the line index
        for j in range(ym, yM+1): # j is the column index
            if ((i-x0)**2 + (j-y0)**2) <= R**2:
                square[lat(j, i)] = onsite

def make_circle2(square, x0, y0, R, onsite):
    xm = int(np.floor(x0-(R+1)))
    xM = int(np.ceil(x0+(R+1)))
    ym = int(np.floor(y0-(R+1)))
    yM = int(np.ceil(y0+(R+1)))
    for i in range(xm, xM+1): # i is the line index
        for j in range(ym, yM+1): # j is the column index
            if R**2 < ((i-x0)**2 + (j-y0)**2) <= (R+1)**2:
                square[lat(j, i)] = onsite

def make_circle_quad(square, x0, y0, R, onsite, iterations, t):
    make_circle1(square, x0, y0, R, onsite)
    n = iterations-1
    b = (4*n*t + 2*n*t*(onsite/t)**0.5)/(onsite - 4*t)
    a = b**2*onsite
    x = np.arange(0, iterations, 1)
    quad_dec = a/(x+b)**2 # quad_dec is the quadratically decreasing potential
    for i in range(1, iterations):
        make_circle2(square, x0, y0, R+i-1, quad_dec[i])

###################### random background potential ############
def make_random_bg_pot(square,W,L) :
    for i in range(L) :
        for j in range(W) :
            if random.randint(0,1) :
                   square[]
W = 100
L = 200
square = scattering(W,L)
square2 = scattering(W,L)
square3 = scattering(W,L)
make_QPC_parabola_quad(square3, W, L, 25, 2.5, 25, 10, 1)
make_circle_quad(square3, x0=W//2, y0=L//3, R=1.5, onsite=25, iterations=5, t=1)
sys = square3.finalized()
mat = np.zeros((W,L))
for i in range(L):
    for j in range(W):
        # print("i,j = {},{}".format(i,j))
        mat[j][i] = square3[lat(i,j)]


##### Current density #####

wfs = kwant.wave_function(sys, energy=0.8)
wf_left = wfs(0)
J0 = kwant.operator.Current(sys)
current = sum(J0(p) for p in wf_left)
kwant.plotter.current(sys, current, cmap='viridis')


##### Plot of the on-site parameter in the lattice #####

fig, ax = plt.subplots()
x = np.arange(0, L, 1)
y = np.arange(0, W, 1)
xx, yy = np.meshgrid(x, y)
ax.scatter(xx, yy)
plt.contourf(x, y, mat)
plt.title("W={},L={},a={},t={}".format(W,L,a,t))
plt.show()

# The function T_of_E is quite heavy and takes some time to execute so I should be careful when 
# calling it with large inputs
# kwant.smatrix().transmission() gives the conductance of the system (see the tutorial section 2.2)

def T_of_E(E_m, E_M, sys, title): # Plots the relation T(E)
    Energy = np.linspace(E_m, E_M, 100)
    Trans = np.zeros(len(Energy))
    for i in tqdm(range(len(Energy))):
        Trans[i] = kwant.smatrix(sys, energy = Energy[i]).transmission(1,0)
        print(str(i)+'/'+str(len(Energy)))
    plt.figure()
    plt.plot(Energy, Trans)
    plt.grid()
    plt.xlabel("Energy [t]")
    plt.ylabel("Conductance [e²/h]")
    plt.title(title)

#T_of_E(0, 5, sys, "W = "+str(W)+" ; L = "+str(L))
#plt.savefig("T_of_E.png")

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