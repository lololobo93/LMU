# -*- coding: utf-8 -*-

"""
	La paquetería empleada en este programa
	se presenta a continuación.
"""

import math                    # http://docs.python.org/library/math.html
from scipy import special	   # https://docs.scipy.org/doc/scipy/
import numpy                   # numpy.scipy.org/
import matplotlib              # matplotlib.sourceforge.net
# mlab y pyplot son funciones numericas y graficas con estilo de MATLAB
import csv
import matplotlib.mlab as mlab # matplotlib.sourceforge.net/api/mlab_api.html
import matplotlib.pyplot as plt# matplotlib.sourceforge.net/api/pyplot_api.html
import matplotlib.cm as cm

"""
   Información de las bobinas empleadas. En este caso se incluyen las 
   medidas de nuestro desacelerador Zeeman.
   Si el usuario lo desea puede descomentar los "input" e introducir las
   dimensiones que desee.
"""

# Se pregunta que sistema se desea simular (Helm=1.0, AntiHelm = -1.0)
T1 = 1.0
# R : Radio de la bobina (m)
R1 = 1.9e-2 #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D1 = 0.0 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro de estas ')
# numz : El número de bobinas enrrolladas en z
numz1 = 68 #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx1 = 28 #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz1 = 1.0e-3 #input('Cual es la distancia entre las bobinas enrrolladas en z ')
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx1 = 1.0e-3 #input('Cual es la distancia entre las bobinas enrrolladas en x ')
# I0 : Corriente recorriendo la bobina (A)
I1 = 2.0

# Corrección para el radio y posición de la bobina
R1 = (R1 + (dwx1/2.0))
D1 = (D1 + (dwz1/2.0))

# Se pregunta que sistema se desea simular.
T2 = T1
# R : Radio de la bobina (m)
R2 = 1.9e-2 #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separacion entre los centros de las bobinas (m)
D2 = 7.0e-2 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro de estas ')
# numz : El número de bobinas enrrolladas en z
numz2 = (48) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx2 = (22) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz2 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx2 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I2 = I1

# Corrección para el radio y posición de la bobina
R2 = (R2 + (dwx2/2.0))
D2 = (D2 + (dwz2/2.0))

# Se pregunta que sistema se desea simular.
T3 = T1
# R : Radio de la bobina (m)
R3 = 1.9e-2 #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D3 = 12.0e-2 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz3 = (48) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx3 = (19) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz3 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx3 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I3 = I1

# Corrección para el radio y posición de la bobina
R3 = (R3 + (dwx3/2.0))
D3 = (D3 + (dwz3/2.0))

# Se pregunta que sistema se desea simular.
T4 = T1
# R : Radio de la bobina (m)
R4 = 1.9e-2 #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D4 = 17.0e-2 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz4 = (48) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx4 = (17) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz4 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx4 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I4 = I1

# Corrección para el radio y posición de la bobina
R4 = (R4 + (dwx4/2.0))
D4 = (D4 + (dwz4/2.0))

# Se pregunta que sistema se desea simular.
T5 = T1
# R : Radio de la bobina (m)
R5 = 1.9e-2 #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D5 = 22.0e-2 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz5 = (48) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx5 = (14) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz5 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx5 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I5 = I1

# Corrección para el radio y posición de la bobina
R5 = (R5 + (dwx5/2.0))
D5 = (D5 + (dwz5/2.0))

# Se pregunta que sistema se desea simular.
T6 = T1
# R : Radio de la bobina (m)
R6 = 1.9e-2 #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D6 = 27.0e-2 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz6 = (48) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx6 = (11) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz6 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx6 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I6 = I1

# Corrección para el radio y posición de la bobina
R6 = (R6 + (dwx6/2.0))
D6 = (D6 + (dwz6/2.0))

# Se pregunta que sistema se desea simular.
T7 = T1
# R : Radio de la bobina (m)
R7 = (1.9e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D7 = 32.0e-2 #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz7 = (38) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx7 = (7) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz7 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx7 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I7 = I1

R7 = (R7 + (dwx7/2.0))
D7 = (D7 + (dwz7/2.0))

# Se pregunta que sistema se desea simular.
T8 = T1
# R : Radio de la bobina (m)
R8 = (1.9e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D8 = (36.0e-2) #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz8 = (33) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx8 = (4) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz8 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx8 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I8 = I1

# Corrección para el radio y posición de la bobina
R8 = (R8 + (dwx8/2.0))
D8 = (D8 + (dwz8/2.0))

# Se pregunta que sistema se desea simular.
T9 = T1
# R : Radio de la bobina (m)
R9 = (1.9e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separacion entre los centros de las bobina (m)
D9 = (44.5e-2) #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El numero de bobinas enrrolladas en z
numz9 = (35) #input('Cuantos alambres tienes en z ')
# numx : El numero de bobinas enrrolladas en x
numx9 = (22) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz9 = dwz1
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx9 = dwx1
# I0 : Corriente recorriendo la bobina (A)
I9 = -1.6

# Corrección para el radio y posición de la bobina
R9 = (R9 + (dwx9/2.0))
D9 = (D9 + (dwz9/2.0))

"""
 Contantes físicas
"""

# Constante de Boltzmann (J/K)
kB = (1.380648e-23)
# uB : Magnetón de Bohr (J/T)
uB = (9.274009 * (10**(-24)))
# Velocidad de la luz (m/s)
c = 3.0*(10**(8))
# Constante de Planck (J*s)
hbar = ((1.054572*10**(-34)))
# Tiempo de vida (s)
tau = (27.2e-9)
# Ancho de línea de la transicion de enfriamiento (1/s)
# Igual para Litio-6 como Litio-7
gamma = 1.0/tau
# Delta de trancisión (1/s)
# Desintonía empleada para enfriamiento
D0 = -66.7*gamma
# Velocidad inicial
Vi = 830.0

"""
   Constantes del sistema.
"""

# Temperatura del horno (K)
Tov = (400.0+273.15)
# Potencia laser (W)
Plas = (60.0e-3)
# Area del laser (m*m)
Alas = math.pi*(((2.54e-2)/2.0)**2)
# Intensidad laser (W/(m*m))
Il = Plas/Alas

"""
   Constante del Litio-6.
"""

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
   Cálculo de características.
   Se determina la distribución de velocidades de los átomos de Litio-6
   y Litio-7 en el horno.
"""

apo2 = numpy.arange(0.0, 5000.0, 1.0) # crea los pasos
fvLi6 = numpy.zeros(apo2.shape) # Aquí se guardarán los resultados para Litio-6 
fvLi7 = numpy.zeros(apo2.shape) # Aquí se guardarán los resultados para Litio-7
len_pasos = len(apo2)

# Cálculo para Litio-6
for i in range(0,len_pasos-1):
	ap1 = (mLi6/(2.0*math.pi*kB*Tov))**3
	ap2 = ap1**(0.5)
	ap3 = 4.0*math.pi*apo2[i]**2
	ap4 = math.exp(-(mLi6*(apo2[i]**2))/(2.0*kB*Tov))
	fvLi6[i] = (ap2*ap3*ap4)

# Cálculo para Litio-7
for i in range(0,len_pasos-1):
	ap1 = (mLi7/(2.0*math.pi*kB*Tov))**3
	ap2 = ap1**(0.5)
	ap3 = 4.0*math.pi*apo2[i]**2
	ap4 = math.exp(-(mLi7*(apo2[i]**2))/(2.0*kB*Tov))
	fvLi7[i] = (ap2*ap3*ap4)

# Velocidad más probable y media para cada átomo
VpLi6 = math.sqrt(3.0*kB*Tov/mLi6)
VmeanLi6 = math.sqrt(8.0*kB*Tov/(math.pi*mLi6))
VpLi7 = math.sqrt(3.0*kB*Tov/mLi7)
VmeanLi7 = math.sqrt(8.0*kB*Tov/(math.pi*mLi7))

# Los átomos con velocidad menor a Vi serán desacelerados en el Zeeman Slower
Vp = Vi

# Se obtienen los campos magnéticos máximos teóricos requeridos para
# desacelerar los átomos con velocidad menor a Vi.
BmaxLi6 = hbar*Vp*kLi6*10000.0/uB
Bmax_primeLi6 = (BmaxLi6+(hbar*(D0)*10000.0/uLi6))

BmaxLi7 = hbar*Vp*kLi7*10000.0/uB
Bmax_primeLi7 = (BmaxLi7+(hbar*(D0)*10000.0/uLi7))

# Longitud mínima para desacelerar los átomos.
lmaxLi6 = Vp**2/(2.0*amaxLi6)
lmax_primeLi6 = ((Vp**2)-((D0)/kLi6)**2)/(2.0*amaxLi6)

lmaxLi7 = Vp**2/(2.0*amaxLi7)
lmax_primeLi7 = ((Vp**2)-((D0)/kLi7)**2)/(2.0*amaxLi7)

# Presentamos resultados

print 'Resultados para Litio-6:'
print 'El valor mas probable de velocidades es (m/s):'
print VpLi6
print 'El valor media de velocidades es (m/s):'
print VmeanLi6
print 'Campo maximo para Litio-6 (G):'
print BmaxLi6
print 'Campo maximo corregida para Litio-6 (G):'
print Bmax_primeLi6
print 'Aceleracion maxima para Litio-6 (m/s*s):'
print amaxLi6
print 'Longitud maxima para Litio-6 (m):'
print lmaxLi6
print 'Longitud maxima corregida para Litio-6 (m):'
print lmax_primeLi6
print 'Velocidad de resonancia para Litio-6 (m/s):'
print (-(D0/kLi6)+(uLi6*Bmax_primeLi6/(10000*hbar*kLi6)))
print ' '
print 'Resultados para Litio-7:'
print 'El valor mas probable de velocidades es (m/s):'
print VpLi7
print 'El valor media de velocidades es (m/s):'
print VmeanLi7
print 'Campo maximo (G):'
print BmaxLi7
print 'Campo maximo corregida (G):'
print Bmax_primeLi7
print 'Aceleracion maxim (m/s*s):'
print amaxLi7
print 'Longitud maxima (m):'
print lmaxLi7
print 'Longitud maxima corregida (m):'
print lmax_primeLi7
print 'Velocidad de resonancia para Li-7 (m/s):'
print (-(D0/kLi7)+(uLi7*Bmax_primeLi7/(10000*hbar*kLi7)))

"""
	Se calculan los perfiles de campo magnético requeridos para 
	desacelerar los átomos de Litio-6 y Litio-7.
"""

apo31 = numpy.arange(0.0, 0.52, 0.001) # crea los pasos
apo3 = numpy.arange(0.0, 0.52, 0.001) # crea los pasos
apo35 = numpy.arange(0.0, 0.56, 0.001) # crea los pasos
apo32 = numpy.arange(0.0, 0.56, 0.001) # crea los pasos

B_teoLi6 = numpy.zeros(apo31.shape) # Aquí se guardarán los resultados
B_teo_primeLi6 = numpy.zeros(apo31.shape) # Aquí se guardarán los resultados
B_teoLi7 = numpy.zeros(apo32.shape) # Aquí se guardarán los resultados
B_teo_primeLi7 = numpy.zeros(apo32.shape) # Aquí se guardarán los resultados

len_pasos1=len(apo31)
len_pasos2=len(apo32)

# A continuación se determinan los perfiles.
for i in range(0,len_pasos2-1):
    ap1 = (kLi7*hbar/uB)
    ap2 = numpy.sqrt((Vp**2)-2.0*amaxLi7*apo32[i])
    B_teoLi7[i] = (ap1 * ap2)

for i in range(0,len_pasos2-1):
    ap1 = (hbar/uB)
    ap2 = kLi7*numpy.sqrt((Vp**2)-2.0*amaxLi7*apo32[i])
    B_teo_primeLi7[i] = (ap1 * (D0+ DelLi7 + ap2))

for i in range(0,len_pasos1-1):
    ap1 = (kLi6*hbar/uB)
    ap2 = numpy.sqrt((Vp**2)-2.0*amaxLi6*apo31[i])
    B_teoLi6[i] = (ap1 * ap2)

for i in range(0,len_pasos1-1):
    ap1 = (hbar/uB)
    ap2 = kLi6*numpy.sqrt((Vp**2)-2.0*amaxLi6*apo31[i])
    B_teo_primeLi6[i] = (ap1 * (D0+ DelLi6 + ap2))

# Correcciones en las distancias.
for i in range(0,len_pasos1-1):
	apo31[i] = apo31[i]-lmax_primeLi6+0.41

for i in range(0,len_pasos2-1):
	apo32[i] = apo32[i]-lmax_primeLi7+0.41

""" 
	Calculamos el campo magnético generado por las bobinas con las 
	dimensiones dadas al comienzo.
	Para ello es necesario definir una función que calcule el campo
	magnético de una espira según las ecuaciones (1) y (2) en 
	Phys. Rev. A Vol. 35, N 4, pp. 1535-1546, 1987.
	
"""

def campos(I,R,D,x,y,z):
    # Lo primero es anotar las funciones de importancia
    # M : Constante de permeabilidad del aire
    M = numpy.float64(4.0 * math.pi * 0.0000001)
    # La siguiente cantidad es una correción en cero
    error = numpy.float64(1e-12)
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
	Aquí se calcula el campo magnético generado por las bobinas con 
	dimensiones introducidas al comienzo.
"""

apo1 = numpy.arange(0.0, 0.52, 0.001) # crea los pasos

# En los siguientes arrrelos se guardarán los resultados.
Bxfin = numpy.zeros(apo1.shape)
Byfin = Bxfin
Bzfin = Bxfin

if (I1 != 0.0):
   for i in range(0, (numz1)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira y el origen
      Dap = D1 + (i*dwz1)
    
      for j in range(0, (numx1)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R1 + (j*dwx1)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I1,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I2 != 0.0):
   for i in range(0, (numz2)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira y el centro
      Dap = D2 + (i*dwz2)
    
      for j in range(0, (numx2)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R2 + (j*dwx2)
   
         # Se calcula el campo magnético de la espira
         [Bx,By,Bz] = campos(I2,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I3 != 0.0):
   for i in range(0, (numz3)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira y el origen
      Dap = D3 + (i*dwz3)
    
      for j in range(0, (numx3)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R3 + (j*dwx3)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I3,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I4 != 0.0):
   for i in range(0, (numz4)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira
      Dap = D4 + (i*dwz4)
    
      for j in range(0, (numx4)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R4 + (j*dwx4)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I4,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I5 != 0.0):
   for i in range(0, (numz5)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira
      Dap = D5 + (i*dwz5)
    
      for j in range(0, (numx5)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R5 + (j*dwx5)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I5,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I6 != 0.0):
   for i in range(0, (numz6)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira
      Dap = D6 + (i*dwz6)
    
      for j in range(0, (numx6)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R6 + (j*dwx6)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I6,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I7 != 0.0):
   for i in range(0, (numz7)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira
      Dap = D7 + (i*dwz7)
    
      for j in range(0, (numx7)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R7 + (j*dwx7)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I7,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I8 != 0.0):
   for i in range(0, (numz8)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira
      Dap = D8 + (i*dwz8)
    
      for j in range(0, (numx8)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R8 + (j*dwx8)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I8,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I9 != 0.0):
   for i in range(0, (numz9)):
      # La siguiente cantidad corresponde a la distancia entre el centro
      # de la espira
      Dap = D9 + (i*dwz9)
    
      for j in range(0, (numx9)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R9 + (j*dwx9)
   
         # Se calcula el campo magnetico de la espira
         [Bx,By,Bz] = campos(I9,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

# Se grafican los resultados obtenidos

# Se retira el factor de corrección al concluir los cálculos.

R1 = (R1 - (dwx1/2.0))
R2 = (R2 - (dwx2/2.0))
R3 = (R3 - (dwx3/2.0))
R4 = (R5 - (dwx4/2.0))
R5 = (R5 - (dwx5/2.0))
R6 = (R6 - (dwx6/2.0))
R7 = (R7 - (dwx7/2.0))
R8 = (R8 - (dwx8/2.0))
R9 = (R9 - (dwx9/2.0))

Sum1=0.0
Sum2=0.0
Sum3=0.0
Sum4=0.0
Sum5=0.0
Sum6=0.0
Sum7=0.0
Sum8=0.0
Sum9=0.0

"""
   Determinación de la distribución de velocidades dentro
   del Slower, para ello se deben de programar las funciones de
   aceleración y el método de Runge Kutta de orden 4.
"""

#Listado de velocidades iniciales
v_init_list=[]
vint=0.0
for i in range(0,49):
    vint = vint+20.0
    v_init_list.append(vint)

# Fuerza de desaceleración para los átomos de Litio-6
def acelLi6(V,B):
    ar1 = hbar*kLi6*gamma/(2.0*mLi6)
    ar2 = ((D0 + (kLi6*V) - ((uLi6/hbar)*B))**2)
    ar3 = (4.0/(gamma**2))
    coci = (1.0 + s0Li6 + (ar3*ar2))
    
    return (-ar1*(s0Li6/coci))

# Fuerza de desaceleración para los átomos de Litio-6
def acelLi7(V,B):
    ar1 = hbar*kLi7*gamma/(2.0*mLi7)
    ar2 = ((D0 + (kLi7*V) - ((uLi7/hbar)*B))**2)
    ar3 = (4.0/(gamma**2))
    coci = (1.0 + s0Li7 + (ar3*ar2))
    
    return (-ar1*(s0Li7/coci))

# Para Litio 6 calculamos el perfil de desaceleración:
# Los resultados se guardarán en los siguiente arreglos.
res_s6 = []
res_v6 = []
res_a6 = []

for v_init in v_init_list:
  v=v_init
  t=0.0
  s=-0.1
  a=0.0
  t_list6=[]
  s_list6=[]
  v_list6=[]    
  a_list6=[]
  res_s6.append(s_list6)
  res_a6.append(a_list6)
  res_v6.append(v_list6)
  while (s<0.6) and (-0.11<=s):
    h=0.000001
    a_list6.append(a)
    s_list6.append(s)
    v_list6.append(v)
    t_list6.append(t)
    if (0.0<=s<0.51):
      # Usamos RUnge-Kutta de cuarto orden
      apx = int(round(s/0.001))
      k1=[v,acelLi6(v,Bzfin[apx])]
      apx = int(round((s+k1[0]*h/2.0)/0.001))
      k2=[v+k1[1]*h/2.0,acelLi6(v+k1[1]*h/2.0,Bzfin[apx])]
      apx = int(round((s+k2[0]*h/2.0)/0.001))
      k3=[v+k2[1]*h/2.0,acelLi6(v+k2[1]*h/2.0,Bzfin[apx])]
      apx = int(round((s+k3[0]*h)/0.001))
      k4=[v+k3[1]*h,acelLi6(v+k3[1]*h,Bzfin[apx])]
      s=s+h/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
      v=v+h/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
      apx = int(round((s)/0.001))
      a=acelLi6(v,Bzfin[apx])
      t=t+h
    else:
      s=s+(v*h)
       
  
# Para Litio 7 calculamos el perfil de desaceleración:
# Los resultados se guardarán en los siguiente arreglos.
res_s7 = []
res_v7 = []
res_a7 = []

for v_init in v_init_list:
  v=v_init
  t=0.0
  s=-0.1
  a=0.0
  t_list7=[]
  s_list7=[]
  v_list7=[]    
  a_list7=[]
  res_s7.append(s_list7)
  res_a7.append(a_list7)
  res_v7.append(v_list7)
  while (s<0.6) and (-0.11<=s):
    h=0.000001
    a_list7.append(a)
    s_list7.append(s)
    v_list7.append(v)
    t_list7.append(t)
    if (0.0<=s<0.51):
       # Usamos RUnge-Kutta de cuarto orden.
       apx = int(round(s/0.001))
       k1=[v,acelLi7(v,Bzfin[apx])]
       apx = int(round((s+k1[0]*h/2.0)/0.001))
       k2=[v+k1[1]*h/2.0,acelLi7(v+k1[1]*h/2.0,Bzfin[apx])]
       apx = int(round((s+k2[0]*h/2.0)/0.001))
       k3=[v+k2[1]*h/2.0,acelLi7(v+k2[1]*h/2.0,Bzfin[apx])]
       apx = int(round((s+k3[0]*h)/0.001))
       k4=[v+k3[1]*h,acelLi7(v+k3[1]*h,Bzfin[apx])]
       s=s+h/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
       v=v+h/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
       t=t+h
       apx = int(round((s)/0.001))
       a=acelLi7(v,Bzfin[apx])
    else:
       s=s+(h*v)

"""
	Ahora determinamos algunas características de las bobinas empleadas
	para el desacelerador Zeeman.
"""

# Se determina la longitud de alambre requerida para construir las bobinas.

if (I1 != 0.0):
   for i in range(1, (numx1)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum1 = Sum1 + (i*dwz1)

   LenghtBob1 = (numz1*2.0*math.pi*((numx1*R1) + Sum1))

if (I2 != 0.0):
   for i in range(1, (numx2)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum2 = Sum2 + (i*dwz2)

   LenghtBob2 = (numz2*2.0*math.pi*((numx2*R2) + Sum2))

if (I3 != 0.0):
   for i in range(1, (numx3)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum3 = Sum3 + (i*dwz3)

   LenghtBob3 = (numz3*2.0*math.pi*((numx3*R3) + Sum3))

if (I1 != 0.0):
   for i in range(1, (numx4)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum4 = Sum4 + (i*dwz4)

   LenghtBob4 = (numz4*2.0*math.pi*((numx4*R4) + Sum4))

if (I2 != 0.0):
   for i in range(1, (numx5)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum5 = Sum5 + (i*dwz5)

   LenghtBob5 = (numz5*2.0*math.pi*((numx5*R5) + Sum5))

if (I3 != 0.0):
   for i in range(1, (numx6)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum6 = Sum6 + (i*dwz6)

   LenghtBob6 = (numz6*2.0*math.pi*((numx6*R6) + Sum6))

if (I1 != 0.0):
   for i in range(1, (numx7)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum7 = Sum7 + (i*dwz7)

   LenghtBob7 = (numz7*2.0*math.pi*((numx7*R7) + Sum7))

if (I2 != 0.0):
   for i in range(1, (numx8)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum8 = Sum8 + (i*dwz8)

   LenghtBob8 = (numz8*2.0*math.pi*((numx8*R8) + Sum8))

if (I3 != 0.0):
   for i in range(1, (numx9)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum9 = Sum9 + (i*dwz9)

   LenghtBob9 = (numz9*2.0*math.pi*((numx9*R9) + Sum9))

LenghtBob = (LenghtBob1 + LenghtBob2 + LenghtBob3)

LenghtBob = LenghtBob+(LenghtBob4 + LenghtBob5 + LenghtBob6)

LenghtBob = LenghtBob+(LenghtBob7 + LenghtBob8 + LenghtBob9)

# Se puede imprimir la longitu de alambre requerido para
# construir las bobinas del desacelerador Zeeman descomentando
# las siguientes dos líneas de código.
#~ print 'La longitud de alambre de las bobinas del desacelerador Zeeman es (m):'
#~ print LenghtBob

# Aquí se determina la potencia disipada por las bobinas.

ResisCu = (1.72 * (10**(-8)))

LenghtBobA=LenghtBob-LenghtBob9

LenghtBobB=LenghtBob9

Area=math.pi*((dwz1/2.0)**2)

# Resistencia de las primeras 8 bobinas
ResA=LenghtBobA*ResisCu/Area

# Resistencia de la última bobina
ResB=LenghtBobB*ResisCu/Area

# Potencia disipada de las primeras 8 bobinas
PotA=ResA*(I1**2)

# Potencia disipada de la última bobina
PotB=ResB*(I9**2)

# Se imprime el resultado
#~ print 'Potencia disipada por el juego de bobinas (W):'
#~ print (PotA+PotB)

"""
	Ahora se grafican los resultados obtenidos en cada sección
"""

plt.figure()
plt.title('Distribucion de velocidades en el horno')
plt.plot(apo2,fvLi6,label="Li6")
plt.plot(apo2,fvLi7,label="Li7")
plt.legend(loc=1)
plt.xlabel('v')
plt.ylabel('f(v)')

cmap = plt.get_cmap('autumn')
colors = [cmap(i) for i in numpy.linspace(0, 1, 48)]

plt.figure()
plt.title('Distribucion de velocidades en el Slower Li-6')
for i, color in enumerate(colors, start=0):
#~ for i in range(0,49):
		plt.plot(res_s6[i],res_v6[i],color=color,linewidth=2)
plt.ylim([-100,1000])
plt.xlim([-0.1,0.6])
plt.xlabel('Posicion [m]',fontsize=20)
plt.ylabel('Velocidad [m/s]',fontsize=20)

plt.figure()
plt.title('Distribucion de velocidades en el Slower Li-7')
for i, color in enumerate(colors, start=0):
		plt.plot(res_s7[i],res_v7[i],color=color,linewidth=2)
plt.ylim([-100,1000])
plt.xlim([-0.1,0.6])
plt.xlabel('Posicion [m]',fontsize=20)
plt.ylabel('Velocidad [m/s]',fontsize=20)

# Comparación entre campo magnético ideal y el obtenido por nuestras bobinas
Bzfin = Bzfin*10000
B_teo_primeLi6= B_teo_primeLi6*10000
B_teo_primeLi7= B_teo_primeLi7*10000

plt.figure()
plt.plot(apo1,Bzfin, label="B calculado")
plt.plot(apo31,B_teo_primeLi6,label="B ideal para Li-6")
plt.plot(apo32,B_teo_primeLi7,label="B ideal para Li-7")
plt.legend(loc=1)
plt.xlim([0.0,0.5])
plt.xlabel('Posicion [cm]',fontsize=20)
plt.ylabel('Campo magnetico [G]',fontsize=20)

#~ plt.figure()
#~ plt.title('Magnitud del campo magnetico (Gauss) en el eje z')
#~ plt.plot(apo3,B_teoLi6,label="Li6")
#~ plt.plot(apo35,B_teoLi7,label="Li7")
#~ plt.legend(loc=1)
#~ plt.xlabel('Eje z (cm)')
#~ plt.ylabel('Eje B (Gauss)')

plt.show()

plt.show()
