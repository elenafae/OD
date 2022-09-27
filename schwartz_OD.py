import numpy as np
from numpy.linalg import inv
from math import cos, sin, pi, radians, sqrt, atan2, asin, acos
import pandas as pd 

mu = 1
k = 0.0172020989484
c = 173.145
theta = radians(23.4379)

#read in file 
df = pd.read_csv("1981QA.txt", sep = " ", names = ["Year", "Month", "Day", "Time", "RA", "DEC", "Sunx", "Suny", "Sunz"])

year = df["Year"].to_numpy()
month = df["Month"].to_numpy()
day = df["Day"].to_numpy()
time = df["Time"].to_numpy()
ra = df["RA"].to_numpy()
dec = df["DEC"].to_numpy()
sunx = df["Sunx"].to_numpy()
suny = df["Suny"].to_numpy()
sunz = df["Sunz"].to_numpy()

R1 = np.array([sunx[0], suny[0], sunz[0]])
R2 = np.array([sunx[1], suny[1], sunz[1]])
R3 = np.array([sunx[2], suny[2], sunz[2]])

#convert to tau 
def ut_to_jd(year, month, day, time):
    t = float(time[:2]) + float(time[3:5])/60 + float(time[6:])/3600
    J0 = 367*year - int((7/4)*(year + int((month+9)/12))) + int(275*month/9) + day + 1721013.5
    JD = J0 + t/24
    return JD

JD1 = ut_to_jd(year[0], month[0], day[0], time[0])
JD2 = ut_to_jd(year[1], month[1], day[1], time[1])
JD3 = ut_to_jd(year[2], month[2], day[2], time[2])
JD_time = [JD1, JD2, JD3]

T3 = k*(JD3 - JD2)
T1 = k*(JD1 - JD2)
T = T3 - T1

#convert ra/dec to radians
def s_to_d(ra, dec):
    a = ra
    d = dec
    #if float(d[:2]) < 0:
        #ra = ((float(a[:2]) + float(a[3:5])/60 + float(a[6:])/3600)*15)*pi/180
        #dec = np.radians((-float(d[1:3]) - float(d[4:6])/60 - float(d[7:])/3600))
        #print("ra, dec", ra, dec)
    #elif float(d[:2]) > 0:
    ra = ((float(a[:2]) + float(a[3:5])/60 + float(a[6:])/3600)*15)*pi/180
    dec = (float(d[1:3]) + float(d[4:6])/60 + float(d[7:])/3600)*pi/180
    print("ra, dec", ra, dec)
    return ra, dec

ra1, dec1 = s_to_d(ra[0],dec[0])
ra2, dec2 = s_to_d(ra[1],dec[1])
ra3, dec3 = s_to_d(ra[2],dec[2])
print("ra, dec", ra1, dec1)
#calculate p hat vectors
phat1 = [cos(ra1)*cos(dec1), sin(ra1)*cos(dec1), sin(dec1)]
phat2 = [cos(ra2)*cos(dec2), sin(ra2)*cos(dec2), sin(dec2)]
phat3 = [cos(ra3)*cos(dec3), sin(ra3)*cos(dec3), sin(dec3)]

#calculate D values 
D0 = np.dot(phat1, np.cross(phat2, phat3))
D11 = np.dot(np.cross(R1, phat2), phat3)
D12 = np.dot(np.cross(R2, phat2), phat3)
D13 = np.dot(np.cross(R3, phat2), phat3)
D21 = np.dot(np.cross(phat1, R1), phat3)
D22 = np.dot(np.cross(phat1, R2), phat3)
D23 = np.dot(np.cross(phat1, R3), phat3)
D31 = np.dot(phat1, np.cross(phat2, R1))
D32 = np.dot(phat1, np.cross(phat2, R2))
D33 = np.dot(phat1, np.cross(phat2, R3))

#create array values for SEL
Ds = [D0, D21, D22, D23]
taus = [T1, T3, T]

def SEL(taus,Sun2,rhohat2,Ds):
    mu = 1 
    A1 = taus[1]/taus[2]
    B1 = (A1/6)*(taus[2]**2 - taus[1]**2)
    A3 = -taus[0]/taus[2]
    B3 = (A3/6)*(taus[2]**2 - taus[0]**2)

    A = (A1*Ds[1] - Ds[2] + A3*Ds[3])/(-Ds[0])
    B = (B1*Ds[1] + B3*Ds[3])/(-Ds[0])
    
    F = np.linalg.norm(Sun2)**2
    E = -2*np.dot(rhohat2, Sun2)
    
    a = -1*(A**2 + A*E + F)
    b = -1*mu*(2*A*B + B*E)
    c = -1*mu**2 * B**2
    
    roots = [1, 0, a, 0, 0, b, 0, 0, c] #for up to three real, positive roots
    roots = np.roots(roots)
    roots2 = []
    rhos = []
    
    for i in range(len(roots)):
        if roots[i] > 0 and np.isreal(roots[i]) == True:
            p2 = A + (mu*B/np.real(roots[i])**3)
            if p2 > 0:
                roots2.append(np.real(roots[i]))
                rhos.append(p2)
    return roots2, rhos

def fg(tau,r2,r2dot,flag):
    r = np.linalg.norm(r2)
    u = mu/r**3
    z = np.dot(r2, r2dot)/r**2
    q = (np.dot(r2dot, r2dot)/r*2) - u
    if flag == "f":
        f = g = 1
    elif flag == "2":
        f = 1 - .5*u*(tau**2)
        g = tau - (1/6)*u*(tau**3)
    elif flag == "3":
        r = np.linalg.norm(r2)
        f = 1 - .5*u*(tau**2) + .5*u*z*(tau**3)
        g = tau - (1/6)*u*(tau**3)
    elif flag == "4":
        f = 1 - .5*u*(tau**2) + .5*u*z*(tau**3) + (1/24)*(3*u*q - 15*u*z**2 + u**2)*(tau**4)
        g = tau - (1/6)*u*(tau**3) + .25*u*z*(tau**4)
    return f, g

def first_it(r2):
    f1, g1 = fg(taus[0], r2, 0, "2")
    f3, g3 = fg(taus[1], r2, 0, "2")

    c1 = g3/(f1*g3 - g1*f3)
    c3 = -g1/(f1*g3 - g1*f3)
    c2 = -1

    p1 = (c1*D11 + c2*D12 +c3*D13)/(c1*D0)
    p2 = (c1*D21 + c2*D22 +c3*D23)/(c2*D0)
    p3 = (c1*D31 + c2*D32 +c3*D33)/(c3*D0)
    p = [p1, p2, p3]
    
    r1 = p1 - R1
    r2 = p2 - R2
    r3 = p3 - R3
    
    d1 = -f3/(f1*g3 - f3*g1)
    d3 = f1/(f1*g3 - f3*g1)

    r2dot = d1*r1 + d3*r3
    return r2, r2dot, p2, p
#first iteration

def sub_it(r2, r2dot, rho2, taus, p, JD_time):
    r_prev = 0
    counter = 1
    p2_now = rho2
    flag = "4" #input("Please enter '3' to truncate to the third term and '4' to truncate to the fourth term:")

    print("Main Iteration Loop:")
    while abs(np.linalg.norm(r2)-r_prev) > 1E-12:
        r_prev = np.linalg.norm(r2)

        f1, g1 = fg(taus[0], r2, r2dot, flag)
        f3, g3 = fg(taus[1], r2, r2dot, flag)
        
        c1 = g3/(f1*g3 - g1*f3)
        c3 = -g1/(f1*g3 - g1*f3)
        c2 = -1
        
        p1 = (c1*D11 + c2*D12 +c3*D13)/(c1*D0)
        p2 = (c1*D21 + c2*D22 +c3*D23)/(c2*D0)
        p3 = (c1*D31 + c2*D32 +c3*D33)/(c3*D0)

        r1 = p1*np.array(phat1) - np.array(R1)
        r2 = p2*np.array(phat2) - np.array(R2)
        r3 = p3*np.array(phat3) - np.array(R3)
    
        d1 = -f3/(f1*g3 - f3*g1)
        d3 = f1/(f1*g3 - f3*g1)

        r2dot = d1*r1 + d3*r3

        p2_prev = p2_now
        p2_now = p2
        delta_p2 = p2_prev - p2_now

        print(f"{counter}: change in rho2 = {delta_p2} au; light-travel time = {p2/c} sec")
        counter += 1

        JD3 = JD_time[2] - p3/c
        JD1 = JD_time[0] - p1/c
        JD2 = JD_time[1] - p2/c

        taus[1] = k*(JD3 - JD2) 
        taus[0] = k*(JD1 - JD2)
        taus[2]  = taus[1] - taus[0] 

    return r2, r2dot, p2
#subsequent iterations

roots,rhos = SEL(taus,R2,phat2,Ds)

def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)

print("Real positive reasonable roots of scalar equation of Lagrange:")

for i in range(len(roots)):
    print(f"({len(roots)}) r2 = {roots[i]} au (rho2 = {rhos[i]} au)")
    r2_arr = []
    r2dot_arr = []

    r2, r2dot, p2, p = first_it(roots[i])
    r2, r2dot, p2 = sub_it(r2, r2dot, p2, taus, p, JD_time)

    print("In x iterations, r2 and r2dot converged to")
    print("r2 = ", r2)
    print("r2_dot = ", r2dot)
    print("in equatorial coordinates")
    print("or")

    rot_matrix = inv(np.array([[1,0,0], [0, cos(theta), sin(theta)], [0, -sin(theta), cos(theta)]]))
    pos = r2@rot_matrix
    v = r2dot@rot_matrix

    print("r2 = ", pos)
    print("r2dot = ", v)
    print("in ecliptic coordinates")
    print(f"with rho2 = {p2} au")

    h = np.cross(pos, v)
    h_mag = np.linalg.norm(h)
    pos_mag = np.linalg.norm(pos)
    v_mag = np.linalg.norm(v)

    a = ((2/pos_mag) - ((v_mag)**2))**-1

    e = (1 - (h_mag**2 / a))**0.5

    I = (acos(h[2]/h_mag)) * 180/pi

    sin_O = h[0]/(h_mag*sin(radians(I)))
    cos_O = -h[1]/(h_mag*sin(radians(I)))

    omega = findQuadrant(sin_O, cos_O)*180/pi

    sin_fw = pos[2]/(pos_mag*sin(radians(I)))
    cos_fw = (1/cos(radians(omega)))*((pos[0]/pos_mag) + (cos(radians(I))*sin_fw*sin_O))

    fw = findQuadrant(sin_fw, cos_fw)

    cos_f = (1/e) * ((a*(1 - e**2)/pos_mag) - 1)
    sin_f = (np.dot(pos, v)/(pos_mag*e))*(sqrt(a*(1-e**2)))

    f = atan2(sin_f, cos_f)

    w = (fw - f)*180/pi


    E = (2*pi) - np.arccos((1/e)*(1 - (pos_mag/a)))

    M = (E - (e*(sin(E)))) * 180/pi

    print("ORBITAL ELEMENTS")
    print("a:", a)
    print("e:", e)
    print("I:", I)
    print("Ω:", omega)
    print("ω:", w)
    print("M:", M)
    print("E:", E*180/pi)
