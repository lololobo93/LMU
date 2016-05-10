# -*- coding: utf-8 -*-

"""
	La paquetería empleada en este programa se presenta a continuación.
"""

import math                    # http://docs.python.org/library/math.html
from scipy import special	   # https://docs.scipy.org/doc/scipy/
import numpy                   # numpy.scipy.org/
import matplotlib              # matplotlib.sourceforge.net
# mlab y pyplot son funciones numericas y graficas con estilo de MATLAB
import matplotlib.mlab as mlab # matplotlib.sourceforge.net/api/mlab_api.html
import matplotlib.pyplot as plt# matplotlib.sourceforge.net/api/pyplot_api.html


"""
	Se introducen las dimensiones de nuestras bobinas de MOT.
"""

# Se pregunta que sistema se desea simular.
T3 = -1.0#input('Que sistema se desea desarrollar (Helmholtz=+1, AntHelmholtz=-1) ')
# R : Radio de la bobina.
R3 = (3.8e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separacion entre los centros de las bobinas.
D3 = (2.605e-2) #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El numero de bobinas enrrolladas en z
numz3 = (6) #input('Cuantos alambres tienes en z ')
# numx : El numero de bobinas enrrolladas en x
numx3 = (4) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z
dwz3 = (4.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en z ')
# dwx : Distancia entre bobinas enrrolladas en x
dwx3 = (1.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en x ')
# I0 : Corriente recorriendo la bobina.
I3 = input('Cual es la corriente en las bobinas (A)')

# Corrección del radio y distancia con el centro de las bobinas de MOT
R3 = (R3 + (dwx3/2.0))
D3 = (D3 + (dwz3/2.0))

"""
	Contantes físicas.
"""

# u : Constante de permeabilidad del aire
u = 4.0 * math.pi * (10**(-7))
# Constante de Boltzmann (J/K)
kB = (1.380648e-23)
# uB : Magneton de Bohr (J/T)
uB = (9.27400968 * (10**(-24)))
# Velocidad de la luz (m/s)
c = 299792458.0
# Constante de Planck (J*s)
hbar = ((1.054571628*10**(-34)))
# Tiempo de vida de la transición del Litio-6 y 7 (s)
tau = (27.102e-9)
# Ancho de linea de la transicion de enfriamiento (1/s)
gamma = (1.0/tau)
# Delta de trancisión de enfriamiento
# Desintonía de los láseres de enfriamiento (1/2)
D0 = -6.0*gamma

"""
   Constantes del sistema.
"""

# Potencia laser (W)
Plas = input('Potencia del laser en Watts: ')
# Area del laser (m2)
Alas = math.pi*(((2.54e-2)/2.0)**2)
# Intensidad laser (W/m2)
Il = Plas/Alas

# mLi6 : Masa del Litio-6 (kg)
mLi6 = (6.0 * (1.660538 * (10**(-27))))
# Numero de onda para Litio-6 (1/m)
kLi6 = 2.0*math.pi/(670.977e-9)
# Longitud de onda del Litio-6 (m)
LLi6 = 670.977e-9
# Intensidad de saturación (W)
IsLi6 = 2.0*(math.pi**2)*hbar*c*gamma/(3.0*(LLi6**3))
# Parámetro de saturación
s0Li6 = Il/IsLi6
# Aceleración máxima (m/s*s)
amaxLi6 = 0.5*(hbar*kLi6*gamma/(2.0*mLi6))*(s0Li6/(1.0+s0Li6))
# Delta Litio-6 (1/s)
# Factor de corrección para un desacelerador spin-flip
DelLi6 = (gamma/2.0)*math.sqrt(1.0+s0Li6)
# Factor de Landé para estado mF=0.5
ggLi6 = 2.00230100
MgLi6 = 0.5
# Factor de Landé para estado mF=1.5
geLi6 = 1.335
MeLi6 = 1.5
# Diferencia de momento magnético entre el estado excitado y base
uLi6 = (MeLi6*geLi6-MgLi6*ggLi6)*uB

"""
   Constantes del Litio-7.
"""

# mLi7 : Masa del Litio-7 (kg)
mLi7 = numpy.float64(7.0 * (1.660538 * (10**(-27))))
# Numero de onda para Litio-7 (1/m)
kLi7 = 2.0*math.pi/(670.962e-9)
# Longitud de onda del Litio-7 (m)
LLi7 = 670.962e-9
# Intensidad de saturación (W)
IsLi7 = 2.0*(math.pi**2)*hbar*c*gamma/(3.0*(LLi7**3))
# Parámetro de saturación
s0Li7 = Il/IsLi7
# Aceleración máxima (m/s*s)
amaxLi7 = 0.5*(hbar*kLi7*gamma/(2.0*mLi7))*(s0Li7/(1.0+s0Li7))
# Delta Litio-7 (1/s)
# Factor de corrección para un desacelerador spin-flip
DelLi7 = (gamma/2.0)*math.sqrt(1+s0Li7)
# Factor de Landé para estado mF=0.5
ggLi7 = 2.00230100
MgLi7 = 0.5
# Factor de Landé para estado mF=1.5
geLi7 = 1.335
MeLi7 = 1.5
# Diferencia de momento magnético entre el estado excitado y base
uLi7 = (MeLi7*geLi7-MgLi7*ggLi7)*uB

""" 
	Se inicia por definir el espacio donde se graficara el 
    campo magnetico.
"""

list1 = [R3, D3]
inter = 1.5e-2#(1.75)*max(list1)# Esto permite una grafica mas completa
delta = 0.0015e-2 # Tamano de pasos en x, y, z, es decir, la resolucion
apo1 = numpy.arange(-inter, inter, delta) # crea los pasos
apo2 = numpy.arange(-inter, inter, delta) # crea los pasos
#X, Z = numpy.meshgrid(apo1, apo2) 
#Y = 0.0

""" 
	A continuacion se escribiran las ecuaciones del campo magnetico
	para una sola bobina, basados en las ecuaciones (1) y (2)
	de Phys Rev A Vol. 35, N 4, pp. 1535-1546, 1987.
	El campo magnetico se obtendra en coordenadas cilindricas (rho,phi,z),
	posteriormente se transcribe a cartesianas.
"""

def campos(I,R,D,x,y,z):
    # Lo primero es anotar las funciones de importancia
    # M : Constante de permeabilidad del aire
    M = 4.0 * math.pi * 0.0000001
    # La siguiente cantidad es una correcion en cero
    error = 1.0e-9
    # rho
    rho = numpy.sqrt(((x)**2 + (y)**2))
    # a1
    a1 = numpy.sqrt((rho + R)**2 + (z - D)**2)
    # k2
    k2 = (4.* R * rho / (a1**2))
    
    # Las siguientes cantidades son parte escencial en las ecuaciones
    # de los campos
    # Funcion K
    K = special.ellipk(k2)
    # Funcion E
    E = special.ellipe(k2)
    # Base para la componente z
    Auxz = ((R**2 - rho**2 - (z - D)**2) / ((R- rho)**2 + (z - D)**2))
    # Base para la componente rho
    Auxr = ((R**2 + rho**2 + (z - D)**2) / ((R- rho)**2 + (z - D)**2))
    
    # A continacion se anotan las ecuaciones de los campos,
    # para ello se emplean las cantidades anteriores
    # El campo en z
    Bz = ((M*I / (2. * math.pi)) * (K + (Auxz * E)) / a1)
    # El campo en rho, el campo en phi es cero
    Brho = ((M*I / (2. * math.pi)) * ((z-D) / (rho+error)) * (-K + (Auxr * E)) / a1)
    
    # Lo siguiente es obtener los resultados del campo en coordenadas
    # cartesianas
    Bx = ((Brho * x) / (rho + error))
    By = ((Brho * y) / (rho + error))
    
    # Finalmente se devuelven los resultados de la funcion
    return [Bx,By,Bz]

""" 
	Paso siguiente, se definen las matrices para obtener los campos
	y, con estas, se determina la magnitud del campo magnetico
	en cada punto del espacio definido por X, Y, Z
"""

# Bobina superior
# La corriente dada por el usuario, se asumira como la corriente en las 
# bobinas superiores
I31 = I3

# Bobina inferior
# La corriente de las bobinas inferiores estara determinada por T
I32 = (I3 * T3)

# Se calcula el campo magnético en el eje z

Bxfin = numpy.zeros(apo1.shape)
Byfin = Bxfin
Bzfin = Bxfin

if (I3 != 0.0):
   for i in range(0, (numz3)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D3 + (i*dwz3)
    
      for j in range(0, (numx3)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R3 + (j*dwx3)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I31,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I32,Rap,-Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

BZtot = numpy.sqrt((Bxfin**2 + Byfin**2 + Bzfin**2))

# El campo y el espacio se convierte en unidades que nos convengan
BZtot = (BZtot * 10000) # El campo se graficará en Gauss

Bxfin = numpy.zeros(apo1.shape)
Byfin = Bxfin
Bzfin = Bxfin

# Se calcula el campo magnético en el eje x

if (I3 != 0.0):
   for i in range(0, (numz3)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D3 + (i*dwz3)
    
      for j in range(0, (numx3)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R3 + (j*dwx3)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I31,Rap,Dap,apo1,0.0,0.0)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I32,Rap,-Dap,apo1,0.0,0.0)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

BXtot = numpy.sqrt((Bxfin**2 + Byfin**2 + Bzfin**2))

# El campo y el espacio se convierte en unidades que nos convengan
BXtot = (BXtot * 10000) # El campo se graficara en Gauss

apo1 = (apo1 * 100) # En cm

Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0

# Se calcula el campo magnético en el origen

if (I3 != 0.0):
   for i in range(0, (numz3)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap3 = D3 + (i*dwz3)
    
      for j in range(0, (numx3)):
        # La siguiente cantidad corresponde al radio de la espira
        Rap3 = R3 + (j*dwx3)
   
        # Se calcula el campo magnetico de la espira superior
        [Bx,By,Bz] = campos(I31,Rap3,Dap3,0.0,0.0,0.0)
        Bx0 = Bx0 + Bx
        By0 = By0 + By
        Bz0 = Bz0 + Bz

        # Se calcula el campo magnetico de la espira inferior
        [Bx,By,Bz] = campos(I32,Rap3,-Dap3,0.0,0.0,0.0)
        Bx0 = Bx0 + Bx
        By0 = By0 + By
        Bz0 = Bz0 + Bz

B0 = numpy.sqrt((Bx0**2 + By0**2 + Bz0**2))

# El campo y el espacio se convierte en unidades que nos convengan
B0 = (B0 * 10000) # El campo se graficara en Gauss

print ' '
print 'El campo magnetico en el centro tiene una magnitud (en Gauss):'
print  B0

""" 
	En lo siguiente se determina la curvatura y gradiente del campo
    magnético en cada direccion (radial y axial).
"""

GradZ = numpy.zeros(apo1.shape)
GradRho = numpy.zeros(apo1.shape)
CurvZ = numpy.zeros(apo1.shape)
CurvRho = numpy.zeros(apo1.shape)

pasos = len(apo1)

for i in range(0, (pasos-1)):
   if i == 0:
	   GradZ[i] = 0.0; GradRho[i] = 0.0; CurvZ[i] = 0.0; CurvRho[i] = 0.0
   elif i == (pasos-1):
	   GradZ[i] = 0.0; GradRho[i] = 0.0; CurvZ[i] = 0.0; CurvRho[i] = 0.0
   else:
	   GradZ[i] = ((BZtot[i+1]-BZtot[i])/(delta*(10.0**2)))
	   GradRho[i] = ((BXtot[i+1]-BXtot[i])/(delta*(10.0**2)))
	   CurvZ[i] = ((BZtot[i+1]-(2.0*BZtot[i])+BZtot[i-1])/((delta**2)*(10.0**4)))
	   CurvRho[i] = ((BXtot[i+1]-(2.0*BXtot[i])+BXtot[i-1])/((delta**2)*(10.0**4)))

print 'El valor de la curvatura axial (eje z), es (en G/cm2):'
print CurvZ[(pasos/2)]
print 'El valor de la curvatura radial, es (en G/cm2):'
print CurvRho[(pasos/2)]
print 'El valor del gradiente axial (eje z), es (en G/cm):'
print abs(GradZ[(pasos/2)-1])
print 'El valor del gradiente radial, es (en G/cm):'
print abs(GradRho[(pasos/2)-1])

"""
   Determinacion del comportamiento de captura de átomos dentro
   de la MOT, para ello se deben de programar las funciones de
   aceleración y el método de Runge Kutta de orden 4.
"""

#Listado de velocidades iniciales para Litio-6
v_init_list=[]
vint=0.0
v_init_list.append(vint)
for i in range(0,25):
    vint = vint+4.0
    v_init_list.append(vint)

#Listado de velocidades iniciales para Litio-7
v_init_list2=[]
vint=0.0
v_init_list.append(vint)
for i in range(0,25):
    vint = vint-4.0
    v_init_list2.append(vint)

# Definicion de las constantes de interes:

# Litio-6
ALi6 = abs(GradRho[(pasos/2)-1])*(1.0/100.0)

betaLi6 = -(8.0*hbar*(kLi6**2)*D0*s0Li6/gamma)
betaLi6 = betaLi6/((1.0+((2.0*D0/gamma)**2))**2)

kappaLi6 = uLi6*betaLi6*ALi6/(hbar*kLi6)

GammaMOT6 = betaLi6/mLi6
OmegaMOT6 = (kappaLi6/mLi6)**(0.5)

Gamma_sc6 = s0Li6*(gamma/2.0)/(1.0+s0Li6)
F_spont6 = hbar*kLi6*Gamma_sc6
R_cap6 = -hbar*D0/(uLi6*ALi6)
VelCap6 = (2.0*R_cap6*F_spont6/mLi6)**0.5

print 'Razon de amortiguamiento del Li6 (Hz):'
print GammaMOT6
print 'Frecuencia de oscilacion del Li6 (Hz):'
print OmegaMOT6
print 'Tiempo de restauracion caracteristico (seg):'
print (2.0*GammaMOT6/(OmegaMOT6**2))
print 'Termino de amortiguamiento:'
print (betaLi6/(2.0*((mLi6*kappaLi6)**0.5)))
print 'Velocidad de captura Li6 (m/s):'
print VelCap6

# Litio-7
ALi7 = abs(GradRho[(pasos/2)-1])*(1.0/100.0)

betaLi7 = -(8.0*hbar*(kLi7**2)*D0*s0Li7/gamma)
betaLi7 = betaLi7/((1.0+s0Li7+((2.0*D0/gamma)**2))**2)

kappaLi7 = uLi7*betaLi7*ALi7/(hbar*kLi7)

GammaMOT7 = betaLi7/mLi7
OmegaMOT7 = (kappaLi7/mLi7)**(0.5)

Gamma_sc7 = s0Li7*(gamma/2.0)/(1.0+s0Li7)
F_spont7 = hbar*kLi7*Gamma_sc7
R_cap7 = -hbar*D0/(uLi7*ALi7)
VelCap7 = (2.0*R_cap7*F_spont7/mLi7)**0.5

print 'Razon de amortiguamiento del Li7 (Hz):'
print GammaMOT7
print 'Frecuencia de oscilacion del Li7 (Hz):'
print OmegaMOT7
print 'Tiempo de restauracion caracteristico (seg):'
print (2.0*GammaMOT7/(OmegaMOT7**2))
print 'Termino de amortiguamiento:'
print (betaLi7/(2.0*((mLi7*kappaLi7)**0.5)))
print 'Velocidad de captura Li7 (m/s):'
print VelCap7

# Fuerza de desaceleración para los átomos de Litio-6
def acelLi6(v,s):
	Fpos = (hbar*kLi6*gamma*s0Li6/2.0)
	DelPos = D0 + (kLi6*v) + (uLi6*ALi6*s/hbar)
	Fpos = Fpos/(1.0+s0Li6+((2.0*DelPos/gamma)**2))
	Fneg = (-hbar*kLi6*gamma*s0Li6/2.0)
	DelNeg = D0 - (kLi6*v) - (uLi6*ALi6*s/hbar)
	Fneg = Fneg/(1.0+s0Li6+((2.0*DelNeg/gamma)**2))
	
	return ((-Fpos-Fneg)/mLi6)

# Fuerza de desaceleración para los átomos de Litio-7
def acelLi7(v,s):
	Fpos = (hbar*kLi7*gamma*s0Li7/2.0)
	DelPos = D0 + (kLi7*v) + (uLi7*ALi7*s/hbar)
	Fpos = Fpos/(1.0+s0Li7+((2.0*DelPos/gamma)**2))
	Fneg = (-hbar*kLi7*gamma*s0Li6/2.0)
	DelNeg = D0 - (kLi7*v) - (uLi7*ALi7*s/hbar)
	Fneg = Fneg/(1.0+s0Li7+((2.0*DelNeg/gamma)**2))
	
	return ((-Fpos-Fneg)/mLi7)

# Para Litio 6 calculamos el perfil de desaceleración:
# Los resultados se guardarán en los siguiente arreglos.
res_s6 = []
res_v6 = []
res_a6 = []

for v_init in v_init_list:
  v=v_init
  t=0.0
  s=-0.0125
  a=0.0
  t_list6=[]
  s_list6=[]
  v_list6=[]    
  a_list6=[]
  res_s6.append(s_list6)
  res_a6.append(a_list6)
  res_v6.append(v_list6)
  h=0.0
  while (s<0.0125) and (t<0.1):
     h=0.000001
     a_list6.append(a)
     s_list6.append(s)
     v_list6.append(v)
     t_list6.append(t)
     apx = s
     apx1=acelLi6(v,apx)
     k1=[v,apx1]
     apx = (s+k1[0]*h/2.0)
     apx1 = acelLi6(v+k1[1]*h/2.0,apx)
     k2 = [v+k1[1]*h/2.0,apx1]
     apx = (s+k2[0]*h/2.0)
     apx1 = acelLi6(v+k2[1]*h/2.0,apx)
     k3 = [v+k2[1]*h/2.0,apx1]
     apx = (s+k3[0]*h)
     apx1 = acelLi6(v+k3[1]*h,apx)
     k4 = [v+k3[1]*h,apx1]
     s = s+h/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
     v = v+h/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
     apx = (s)
     a=acelLi6(v,apx)
     t=t+h
  
# Para Litio 6 calculamos el perfil de desaceleración:
# Los resultados se guardarán en los siguiente arreglos.
res_s7 = []
res_v7 = []
res_a7 = []

for v_init in v_init_list:
  v=v_init
  t=0.0
  s=-0.0125
  a=0.0
  t_list7=[]
  s_list7=[]
  v_list7=[]    
  a_list7=[]
  res_s7.append(s_list7)
  res_a7.append(a_list7)
  res_v7.append(v_list7)
  h=0.0
  while (s<0.0125) and (t<0.1):
    h=0.000001
    a_list7.append(a)
    s_list7.append(s)
    v_list7.append(v)
    t_list7.append(t)
    apx = (s)
    apx1 = acelLi7(v,apx)
    k1=[v,apx1]
    apx = (s+k1[0]*h/2.0)
    apx1 = acelLi7(v+k1[1]*h/2.0,apx)
    k2=[v+k1[1]*h/2.0,apx1]
    apx = (s+k2[0]*h/2.0)
    apx1 = acelLi7(v+k2[1]*h/2.0,apx)
    k3=[v+k2[1]*h/2.0,apx1]
    apx = (s+k3[0]*h)
    apx1 = acelLi7(v+k3[1]*h,apx)
    k4=[v+k3[1]*h,apx1]
    s=s+h/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
    v=v+h/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
    t=t+h
    apx = (s)
    a=acelLi7(v,apx)

# Velocidades en la otra dirección

# Para Litio 6 calculamos el perfil de desaceleración:
# Los resultados se guardarán en los siguiente arreglos.
res_s62 = []
res_v62 = []
res_a62 = []

for v_init in v_init_list2:
  v=v_init
  t=0.0
  s=0.0125
  a=0.0
  t_list62=[]
  s_list62=[]
  v_list62=[]    
  a_list62=[]
  res_s62.append(s_list62)
  res_a62.append(a_list62)
  res_v62.append(v_list62)
  h=0.0
  while (s>-0.0125) and (t<0.1):
     h=0.00001
     a_list62.append(a)
     s_list62.append(s)
     v_list62.append(v)
     t_list62.append(t)
     apx = s
     apx1=acelLi6(v,apx)
     k1=[v,apx1]
     apx = (s+k1[0]*h/2.0)
     apx1=acelLi6(v+k1[1]*h/2.0,apx)
     k2=[v+k1[1]*h/2.0,apx1]
     apx = (s+k2[0]*h/2.0)
     apx1 = acelLi6(v+k2[1]*h/2.0,apx)
     k3=[v+k2[1]*h/2.0,apx1]
     apx = (s+k3[0]*h)
     apx1= acelLi6(v+k3[1]*h,apx)
     k4=[v+k3[1]*h,apx1]
     s=s+h/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
     v=v+h/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
     apx = (s)
     a=acelLi6(v,apx)
     t=t+h
  
# Para Litio 7 calculamos el perfil de desaceleración:
# Los resultados se guardarán en los siguiente arreglos.
res_s72 = []
res_v72 = []
res_a72 = []

for v_init in v_init_list2:
  v=v_init
  t=0.0
  s=0.0125
  a=0.0
  t_list72=[]
  s_list72=[]
  v_list72=[]    
  a_list72=[]
  res_s72.append(s_list72)
  res_a72.append(a_list72)
  res_v72.append(v_list72)
  h=0.0
  while (s>-0.0125) and (t<0.1):
    h=0.00001
    a_list72.append(a)
    s_list72.append(s)
    v_list72.append(v)
    t_list72.append(t)
    apx = (s)
    apx1 = acelLi7(v,apx)
    k1=[v,apx1]
    apx = (s+k1[0]*h/2.0)
    apx1 = acelLi7(v+k1[1]*h/2.0,apx)
    k2=[v+k1[1]*h/2.0,apx1]
    apx = (s+k2[0]*h/2.0)
    apx1 = acelLi7(v+k2[1]*h/2.0,apx)
    k3=[v+k2[1]*h/2.0,apx1]
    apx = (s+k3[0]*h)
    apx1 = acelLi7(v+k3[1]*h,apx)
    k4=[v+k3[1]*h,apx1]
    s=s+h/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
    v=v+h/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
    t=t+h
    apx = (s)
    a=acelLi7(v,apx)

# Aquí presentamos la región de captura, la cual también es graficada
# con los resultados anteriores.

def v1Li6(z):
	Apx = (-D0-(uLi6*ALi6*z/hbar))/kLi6
	return Apx

def v2Li6(z):
	Apx = (D0-(uLi6*ALi6*z/hbar))/kLi6
	return Apx

def v1Li7(z):
	Apx = (-D0-(uLi7*ALi7*z/hbar))/kLi7
	return Apx

def v2Li7(z):
	Apx = (D0-(uLi7*ALi7*z/hbar))/kLi7
	return Apx

# Intervalor de captura
z_init=[]
zint=-0.0125
z_init.append(zint)
for i in range(0,500):
    zint = zint+0.0005
    z_init.append(zint)

# Aquí se guardan los resultados
v_linea16 = []
v_linea26 = []
v_linea17 = []
v_linea27 = []

for z_ex in z_init:
  z=z_ex
  v_linea16.append(v1Li6(z))
  v_linea26.append(v2Li6(z))
  v_linea17.append(v1Li7(z))
  v_linea27.append(v2Li7(z))

"""
	Los resultados obtenidos se grafican finalmente.
"""

plt.figure()
plt.title('Distribucion de velocidades en MOT, Litio-6')
for i in range(0,26):
    plt.plot(res_s6[i],res_v6[i],color="blue")
for i in range(0,25):
    plt.plot(res_s62[i],res_v62[i],color="green")
plt.plot(z_init,v_linea16,color="red")
plt.plot(z_init,v_linea26,color="red")
plt.ylim([-100,100])
plt.xlim([-0.0125,0.0125])
plt.legend(loc=1)
plt.xlabel('Posicion [m]',fontsize=20)
plt.ylabel('Velocidad [m/s]',fontsize=20)

plt.figure()
plt.title('Distribucion de velocidades en MOT, Litio-7')
for i in range(0,26):
    plt.plot(res_s7[i],res_v7[i],color="blue")
for i in range(0,25):
    plt.plot(res_s72[i],res_v72[i],color="green")
plt.plot(z_init,v_linea16,color="red")
plt.plot(z_init,v_linea26,color="red")
plt.ylim([-100,100])
plt.xlim([-0.0125,0.0125])
plt.legend(loc=1)
plt.xlabel('Posicion [m]',fontsize=20)
plt.ylabel('Velocidad [m/s]',fontsize=20)

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje z')
plt.plot(apo1,BZtot)
plt.xlabel('Eje z (cm)')
plt.ylabel('Eje B (Gauss)')

Bxfin=Bxfin*10000.0

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje x')
plt.plot(apo1,BXtot)
plt.xlabel('Eje x (cm)')
plt.ylabel('Eje B (Gauss)')

plt.show()
