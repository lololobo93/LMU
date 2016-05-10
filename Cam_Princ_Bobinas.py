# -*- coding: utf-8 -*-

"""
	La paquetería empleada en este programa se presenta a continuación
"""

import math                    # http://docs.python.org/library/math.html
from scipy import special	   # https://docs.scipy.org/doc/scipy/
import numpy                   # numpy.scipy.org/
import matplotlib              # matplotlib.sourceforge.net
# mlab y pyplot son funciones numericas y graficas con estilo de MATLAB
import matplotlib.mlab as mlab # matplotlib.sourceforge.net/api/mlab_api.html
import matplotlib.pyplot as plt# matplotlib.sourceforge.net/api/pyplot_api.html
from mpl_toolkits.mplot3d import Axes3D # Imágenes 3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter # Imágenes 3D
import matplotlib.cm as cm # Imágenes 3D

""" Todos los parametros que se piden a continuacion deben
	ingresarse como flotantes en unidades del SI, i.e. metro,
	Ampere, Tesla. El eje z es definido como el eje de las bobinas
	El punto medio entre las bobinas es tomado como el origen 
	(x,y,z)=(0,0,0)
"""

print 'En este programa es posible introducir hasta tres bobinas.'
numbob = input('Cuantos pares de bobinas quieres simular (maximo 3): ')
print 'Introduzca las especificaciones de las bobinas.'
print 'Introduzca estos valores como reales, en unidades del SI.'
print 'ATENCION: Las dimensiones que solicita el programa corresponden'
print 'al punto donde inician las BOBINAS (no el centro de los alambres).'
print ' '

"""
	Se introducen las dimensiones de nuestras bobinas de Feshbach y MOT.
	Las bobinas de Feshbach están divididas en dos partes, por lo cual
	se tienen tres juegos de bobinas.
"""

# Es necesario escribir algunos valores iniciales, 
# para las variables que no se empleen.

# Se pregunta que sistema se desea simular.
T2 = 0.0
# R : Radio de la bobina.
R2 = 0.0
# D : Separación entre los centros de las bobinas.
D2 = 0.0
# numz : El número de bobinas enrrolladas en z
numz2 = 0.0
# numx : El número de bobinas enrrolladas en x
numx2 = 0.0
# dwz : Distancia entre bobinas enrrolladas en z
dwz2 = 0.0
# dwx : Distancia entre bobinas enrrolladas en x
dwx2 = 0.0
# I0 : Corriente recorriendo la bobina.
I2 = 0.0

# Se pregunta que sistema se desea simular.
T3 = 0.0
# R : Radio de la bobina.
R3 = 0.0
# D : Separación entre los centros de las bobinas.
D3 = 0.0
# numz : El número de bobinas enrrolladas en z
numz3 = 0.0
# numx : El número de bobinas enrrolladas en x
numx3 = 0.0
# dwz : Distancia entre bobinas enrrolladas en z
dwz3 = 0.0
# dwx : Distancia entre bobinas enrrolladas en x
dwx3 = 0.0
# I0 : Corriente recorriendo la bobina.
I3 = 0.0

# Bobinas de Feshbach pequeñas

print 'Valores de la primera bobina.'
# Se pregunta que sistema se desea simular.
T1 = input('Que sistema se desea desarrollar (Helmholtz=+1, AntHelmholtz=-1) ')
# R : Radio de la bobina (m)
R1 = (8.1e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
# D : Separación entre los centros de las bobinas (m)
D1 = (4.2e-2) #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
# numz : El número de bobinas enrrolladas en z
numz1 = (2) #input('Cuantos alambres tienes en z ')
# numx : El número de bobinas enrrolladas en x
numx1 = (6) #input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z (m)
dwz1 = (4.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en z ')
# dwx : Distancia entre bobinas enrrolladas en x (m)
dwx1 = (4.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en x ')
# I0 : Corriente recorriendo la bobina (A)
I1 = input('Cual es la corriente en las bobinas (positiva) ')

# Corrección del radio y altura de las bobinas.
R1 = (R1 + (dwx1/2.0))
D1 = (D1 + (dwz1/2.0))

# Bobinas de Feshbach grandes

if (numbob>1.5):
 print 'Valores de la segunda bobina.'
 # Se pregunta que sistema se desea simular.
 T2 = input('Que sistema se desea desarrollar (Helmholtz=+1, AntHelmholtz=-1) ')
 # R : Radio de la bobina (m)
 R2 = (8.1e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
 # D : Separación entre los centros de las bobinas (m)
 D2 = (5.12e-2) #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
 # numz : El número de bobinas enrrolladas en z
 numz2 = (10) #input('Cuantos alambres tienes en z ')
 # numx : El número de bobinas enrrolladas en x
 numx2 = (6) #input('Cuantos alambres tienes en x ')
 # dwz : Distancia entre bobinas enrrolladas en z (m)
 dwz2 = (4.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en z ')
 # dwx : Distancia entre bobinas enrrolladas en x (m)
 dwx2 = (4.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en x ')
 # I0 : Corriente recorriendo la bobina (A)
 I2 = input('Cual es la corriente en las bobinas (positiva) ')

 # Corrección del radio y altura de las bobinas.
 R2 = (R2 + (dwx2/2.0))
 D2 = (D2 + (dwz2/2.0))

# Bobinas de MOT

if (numbob>2.5):
 print 'Valores de la tercera bobina.'
 # Se pregunta que sistema se desea simular.
 T3 = input('Que sistema se desea desarrollar (Helmholtz=+1, AntHelmholtz=-1) ')
 # R : Radio de la bobina (m)
 R3 = (3.8e-2) #input('Cual es el radio mas pequeno de las bobinas  ')
 # D : Separación entre los centros de las bobinas (m)
 D3 = (2.605e-2) #input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas ')
 # numz : El numero de bobinas enrrolladas en z
 numz3 = (6) #input('Cuantos alambres tienes en z ')
 # numx : El numero de bobinas enrrolladas en x
 numx3 = (4) #input('Cuantos alambres tienes en x ')
 # dwz : Distancia entre bobinas enrrolladas en z (m)
 dwz3 = (4.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en z ')
 # dwx : Distancia entre bobinas enrrolladas en x (m)
 dwx3 = (1.6e-3) #input('Cual es la distancia entre las bobinas enrrolladas en x ')
 # I0 : Corriente recorriendo la bobina (A)
 I3 = input('Cual es la corriente en las bobinas (positiva) ')

 # Corrección del radio y altura de las bobinas.
 R3 = (R3 + (dwx3/2.0))
 D3 = (D3 + (dwz3/2.0))

# alpha : Pide el gruso del kapton del alambre
alpha = (0.3e-3) #input('Cual es el grosor del alambre ')

"""
	Constante físicas
"""

# u : Constante de permeabilidad del aire
u = 4.0 * math.pi * (10**(-7))
# uB : Magneton de Bohr (J/T)
uB = numpy.float64(9.274009 * (10**(-24)))
# mLi : Masa del Litio-6 (kg)
mLi = (6.0 * (1.660538 * (10**(-27))))

""" 
	A continuación se calcula el campo magnético producido por el sistema
	de bobinas.
"""

list1 = [R1, D1, R2, D2, R3, D3]
inter = 1.5e-2#(1.75)*max(list1)# Esto permite una gráfica mas completa.
# Se determina la región donde se graficará el campo magnético.
delta = 0.01e-2 # Tamano de pasos en x, y, z, es decir, la resolución
apo1 = numpy.arange(-inter, inter, delta) # crea los pasos
apo2 = numpy.arange(-inter, inter, delta) # crea los pasos
X, Z = numpy.meshgrid(apo1, apo2) 
Y = 0.0

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
	Para determinar el momento magnético de cada estado del Litio-6 se 
	empleará la siguiente función basada en la fórmula de Breit-Rabi.
"""

def MgLi6(mF , B):
    # Lo primero es anotar las cantidades de importancia
    # uBHz: Magneton de Bohr en unidades de MHz/G, el cual es de suma
    # importancia en los calculos del momento magnetico del Litio-6
    uBHz = 1.399624624
    # ILi6: Valor del numero cuantico I para el Litio-6
    ILi6 = 1.0
    # gILi6: Factor de Lande I para el Litio-6
    gILi6 = -0.000447654
    # gJLi6: Factor de Lande J para el Litio-6
    gJLi6 = 2.0023010
    # AhfLi6: Factor hiperfino entre los niveles de energia en MHz
    AhfLi6 = 152.1368407
    # EhfLi6: Diferencia de enrgia entre los niveles hiperfinos del Litio-6
    EhfLi6 = (AhfLi6 * (ILi6 + 0.5))
    
    # A continuacion se anotan algunas cantidades que simplificaran la funcion:
    b1 = (gILi6 * uBHz * mF)
    b2 = (1./4.) * EhfLi6
    b3 = (4.0 * mF * (gJLi6 - gILi6) * uBHz)
    b4 = (1.0 / (((2.0*ILi6) + 1.0) * EhfLi6))
    b5 = (((gJLi6 - gILi6)**2) * (uBHz**2) / (EhfLi6**2))
    b6 = (b3 * b4)
    b7 = (b6 * B)
    b8 = (2.0 * b5 * B)
    b9 = (b5 * (B**2))
    b10 = (b2 * (b6 + b8))
    b11 = numpy.sqrt(1.0 + b7 + b9)
    # El resultado final estara dado por:
    
    MgLi6 = (b1 - (b10 / b11))
    
    # Finalmente se devuelven los resultados de la funcion
    return MgLi6

""" 
	Paso siguiente, se definen las matrices para obtener los campos
	y, con estas, se determina la magnitud del campo magnetico
	en cada punto del espacio definido por X, Y, Z
"""

"""
	Se procede al cálculo del campo magnético generado por el conjunto
	de bobinas.
"""

# Bobinas superiores
# La corriente dada por el usuario, se asumirá como la corriente en las 
# bobinas superiores
I11 = I1
I21 = I2
I31 = I3

# Bobinas inferiores
# La corriente de las bobinas inferiores estará determinada por T
I12 = (I1 * T1)
I22 = (I2 * T2)
I32 = (I3 * T3)
      
# En las siguientes matrices se almanecerán
# los resultados del campo magnetico en x, y y z
Bxfin = numpy.zeros(X.shape)
Byfin = Bxfin
Bzfin = Bxfin

# Se calcula el campo magnético en el plano x-z

if (I1 != 0.0):
   for i in range(0, (numz1)):
#      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D1 + (i*dwz1)
#    
      for j in range(0, (numx1)):
#         # La siguiente cantidad corresponde al radio de la espira
         Rap = R1 + (j*dwx1)
   
#         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I11,Rap,Dap,X,Y,Z)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior y se suma
         # con el campo obtenido para la espira superior
         [Bx,By,Bz] = campos(I12,Rap,-Dap,X,Y,Z)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I2 != 0.0):
   for i in range(0, (numz2)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D2 + (i*dwz2)
    
      for j in range(0, (numx2)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R2 + (j*dwx2)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I21,Rap,Dap,X,Y,Z)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I22,Rap,-Dap,X,Y,Z)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I3 != 0.0):
   for i in range(0, (numz3)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D3 + (i*dwz3)
    
      for j in range(0, (numx3)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R3 + (j*dwx3)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I31,Rap,Dap,X,Y,Z)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnético de la espira inferior
         [Bx,By,Bz] = campos(I32,Rap,-Dap,X,Y,Z)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

# Se calcula la magnitud del campo magnético obtenido anteriormente
B1 = numpy.sqrt((Bxfin**2 + Byfin**2 + Bzfin**2))
# El campo y el espacio se convierte en unidades que nos convengan
B1 = (B1 * 10000) # El campo se graficara en Gauss

X = (X * 100) # En cm
Y = (Y * 100) # En cm
Z = (Z * 100) # En cm

Bxfin = numpy.zeros(apo1.shape)
Byfin = Bxfin
Bzfin = Bxfin

# Ahora se procede a calcular el campo magnético en el eje z

if (I1 != 0.0):
   for i in range(0, (numz1)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D1 + (i*dwz1)
    
      for j in range(0, (numx1)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R1 + (j*dwx1)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I11,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I12,Rap,-Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I2 != 0.0):
   for i in range(0, (numz2)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D2 + (i*dwz2)
    
      for j in range(0, (numx2)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R2 + (j*dwx2)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I21,Rap,Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I22,Rap,-Dap,0.0,0.0,apo1)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

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

# Igualmente se calcula el campo magnético en el eje x.

if (I1 != 0.0):
   for i in range(0, (numz1)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D1 + (i*dwz1)
    
      for j in range(0, (numx1)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R1 + (j*dwx1)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I11,Rap,Dap,apo1,0.0,0.0)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I12,Rap,-Dap,apo1,0.0,0.0)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

if (I2 != 0.0):
   for i in range(0, (numz2)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap = D2 + (i*dwz2)
    
      for j in range(0, (numx2)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap = R2 + (j*dwx2)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I21,Rap,Dap,apo1,0.0,0.0)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I22,Rap,-Dap,apo1,0.0,0.0)
         Bxfin = Bxfin + Bx
         Byfin = Byfin + By
         Bzfin = Bzfin + Bz

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

# Finalmente se calcula el campo magnético en el origen

if (I1 != 0.0):
   for i in range(0, (numz1)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap1 = D1 + (i*dwz1)
    
      for j in range(0, (numx1)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap1 = R1 + (j*dwx1)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I11,Rap1,Dap1,0.0,0.0,0.0)
         Bx0 = Bx0 + Bx
         By0 = By0 + By
         Bz0 = Bz0 + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I12,Rap1,-Dap1,0.0,0.0,0.0)
         Bx0 = Bx0 + Bx
         By0 = By0 + By
         Bz0 = Bz0 + Bz

if (I2 != 0.0):
   for i in range(0, (numz2)):
      # La siguiente cantidad corresponde a la distancia entre el centro
#      # de la espira y el centro
      Dap2 = D2 + (i*dwz2)
    
      for j in range(0, (numx2)):
         # La siguiente cantidad corresponde al radio de la espira
         Rap2 = R2 + (j*dwx2)
   
         # Se calcula el campo magnetico de la espira superior
         [Bx,By,Bz] = campos(I21,Rap2,Dap2,0.0,0.0,0.0)
         Bx0 = Bx0 + Bx
         By0 = By0 + By
         Bz0 = Bz0 + Bz

         # Se calcula el campo magnetico de la espira inferior
         [Bx,By,Bz] = campos(I22,Rap2,-Dap2,0.0,0.0,0.0)
         Bx0 = Bx0 + Bx
         By0 = By0 + By
         Bz0 = Bz0 + Bz

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

""" 
	El programa además calcula la frecuencia de confinamiento
    en la direccion axial para cada estado mF del atomo de Litio-6.
    Como primera parte se debe de calcular el momento magnético de cada estado
    de la estructura hiperfina del Litio-6, es decir en los estados F=1/2,1/2
    y F=1/2,-1/2, para lo cual se contruyó al función MgLi6.
"""

MgLi61 = MgLi6(-0.5 , B0)  # Para el estado mF = -1/2, ademas depende del valor del campo aplicado
MgLi62 = MgLi6(0.5 , B0) # Para el estado mF = 1/2, ademas depende del valor del campo aplicado

# Lo siguiente es calcular el valor de la frecuencia en cada direccion, se tendra:

CR = abs(CurvRho[(pasos/2)])
CZ = abs(CurvZ[(pasos/2)])
ud = abs(MgLi61)
uu = abs(MgLi62)

w1r = ((1.0/(2.0*math.pi))*((CR*ud*uB/mLi)**(0.5))) # Frecuencia radial en el estado mF = -1/2
w2r = ((1.0/(2.0*math.pi))*((CR*uu*uB/mLi)**(0.5))) # Frecuencia radial en el estado mF = 1/2
w1z = ((1.0/(2.0*math.pi))*((CZ*ud*uB/mLi)**(0.5))) # Frecuencia axial en el estado mF = -1/2
w2z = ((1.0/(2.0*math.pi))*((CZ*uu*uB/mLi)**(0.5))) # Frecuencia axial en el estado mF = 1/2

print 'El valor de la curvatura axial (eje z), es (en G/cm2):'
print CurvZ[(pasos/2)]
print 'El valor de la curvatura radial, es (en G/cm2):'
print CurvRho[(pasos/2)]
print 'El valor del gradiente axial (eje z), es (en G/cm):'
print abs(GradZ[(pasos/2)-1])
print 'El valor del gradiente radial, es (en G/cm):'
print abs(GradRho[(pasos/2)-1])
print 'Frecuencia axial en el estado mF = -1/2 (en Hz):'
print (numpy.sign(CurvZ[(pasos/2)])*(-1)*w1z)
print 'Frecuencia axial en el estado mF = 1/2 (en Hz):'
print (numpy.sign(CurvZ[(pasos/2)])*(-1)*w2z)
print 'Frecuencia radial en el estado mF = -1/2 (en Hz):'
print (numpy.sign(CurvRho[(pasos/2)])*(-1)*w1r)
print 'Frecuencia radial en el estado mF = 1/2 (en Hz):'
print (numpy.sign(CurvRho[(pasos/2)])*(-1)*w2r)

""" 
	En lo siguiente se determina el valor de la inductancia para 
    cada una de las bobinas.
"""

R1 = (R1 - (dwx1/2.0))
D1 = (D1 - (dwz1/2.0))
R2 = (R2 - (dwx2/2.0))
D2 = (D2 - (dwz2/2.0))
R3 = (R3 - (dwx3/2.0))
D3 = (D3 - (dwz3/2.0))

Induc1 = 0.0
Induc2 = 0.0
Induc3 = 0.0

if (I1 != 0.0):
   # Variables de apoyo para calcular la inductancia
   L11 = R1
   L12 = R1 + (dwx1*numx1)
   leng = ((numz1)*dwz1)

   # Se determina la inductancia
   ap1 = (31.6 * (L11**2) * ((numx1*numz1)**2))
   ap2 = ((6.0*L11) + (9.0*leng) + 10.0*(L12-L11))
   Induc1 = 2.0*ap1/ap2

if (I2 != 0.0):
   # Variables de apoyo para calcular la inductancia
   L21 = R2
   L22 = R2 + (dwx2*numx2)
   leng = ((numz2)*dwz2)

   # Se determina la inductancia
   ap1 = (31.6 * (L21**2) * ((numx2*numz2)**2))
   ap2 = (6.0*L21 + 9.0*leng + 10.0*(L22-L21))
   Induc2 = 2.0*ap1/ap2

if (I3 != 0.0):
   # Variables de apoyo para calcular la inductancia
   L31 = R3
   L32 = R3 + (dwx3*numx3)
   leng = ((numz3)*dwz3)

   # Se determina la inductancia
   ap1 = (31.6 * (L31**2) * ((numx3*numz3)**2))
   ap2 = (6.0*L31 + 9.0*leng + 10.0*(L32-L31))
   Induc3 = ap1/ap2

# Se indica el valor de la inductancia en pantalla
if (I1 != 0.0):
  print 'El valor de la inductancia para una de las bobinas del primer juego es (en uH):'
  print Induc1

if (I2 != 0.0):
  print 'El valor de la inductancia para una de las bobinas del segundo juego es (en uH):'
  print Induc2

if (I3 != 0.0):
  print 'El valor de la inductancia para una de las bobinas del tercer juego es (en uH):'
  print Induc3

""" 
	Aprovechando lo anterior se realizara el calculo de la longitud de 
    alambre necesario para construir esta bobina, ademas de determinar 
    la potencia disipada por la bobina en cuestion.
"""

# Es necesario definir algunas constantes:
# CondTermAirMet: Conductividad termica del aire
CondTermAirMet = (10.0 * (10**(-6)))
# CondTermAguMet: Conductividad termica del agua
CondTermAguMet = (500.0 * (10**(-6)))
# ResisCu : Resistividad del cobre (Ohms por metro)
ResisCu = (1.71 * (10**(-8)))

# Se procede a calcular lo solicitado:
ResisBob1 = 0.0
ResisBob2 = 0.0
ResisBob3 = 0.0

if (I1 != 0.0):
   ResisBob1 = (ResisCu * math.pi * (2.0*R1 + (dwx1*numx1)) * (numx1 * numz1))
   ResisBob1 = 2.0 * ResisBob1 / ((dwx1*dwz1)-((2.0e-3)**2))
   print 'El valor de la resistencia para una de las bobinas del primer juego es (en Ohms):'
   print ResisBob1
   print 'El voltaje necesario para alimentar esta bobina es (V):'
   print (ResisBob1*I1)
   print 'El tiempo de descarga de esta bobina es (ms):'
   print (5.0*Induc1/(ResisBob1*1000.0))
if (I2 != 0.0):
   ResisBob2 = (ResisCu * math.pi * (2.0*R2 + (dwx2*numx2)) * (numx2 * numz2))
   ResisBob2 = 2.0 * ResisBob2 / ((dwx1*dwz1)-((2.0e-3)**2))
   print 'El valor de la resistencia para una de las bobinas del segundo juego es (en Ohms):'
   print ResisBob2
   print 'El voltaje necesario para alimentar esta bobina es (V):'
   print (ResisBob2*I2)
   print 'El tiempo de descarga de esta bobina es (ms):'
   print (5.0*Induc2/(ResisBob2*1000.0))
if (I3 != 0.0):
   ResisBob3 = (ResisCu * math.pi * (2.0*R3 + (dwx3*numx3)) * (numx3 * numz3))
   ResisBob3 = 2.0 * ResisBob3 / ((dwx3) * (dwz3))
   print 'El valor de la resistencia para una de las bobinas del tercer juego es (en Ohms):'
   print ResisBob3
   print 'El voltaje necesario para alimentar esta bobina es (V):'
   print (ResisBob3*abs(I3))
   print 'El tiempo de descarga de esta bobina es (ms):'
   print (5.0*Induc3/(ResisBob3*1000.0))

# Lo anterior correspondio al calculo de la resistividad de la bobina,
# mientras la potencia disipada estara dada por:

PotDisBob1 = (2.0*(ResisBob1 * (I1**2)))
PotDisBob2 = (2.0*(ResisBob2 * (I2**2)))
PotDisBob3 = (2.0*(ResisBob3 * (I3**2)))

# Sin embargo, la anterior es la potencia disipada para una sola bobina
# para las dos bobinas se tendra:

PotDisTot = (PotDisBob1 + PotDisBob1 + PotDisBob1)

print 'La potencia disipada por las bobinas es (en Watts):'
print PotDisTot

# Procedemos a calcular la longitud de alambre necesaria para construir
# estas bobinas.

Sum1 = 0.0
Sum2 = 0.0
Sum3 = 0.0

LenghtBob1 = 0.0
LenghtBob2 = 0.0
LenghtBob3 = 0.0

if (I1 != 0.0):
   for i in range(1, (numx1)):
      # En lo siguiente se determina la longitud de espiras en la direccion radial
      Sum1 = Sum1 + (i*dwz1)

   LenghtBob1 = (2.0*numz1*2.0*math.pi*((numx1*L11) + Sum1))

if (I2 != 0.0):
   for i in range(1, (numx2)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum2 = Sum2 + (i*dwz2)

   LenghtBob2 = (2.0*numz2*2.0*math.pi*((numx2*L21) + Sum2))

if (I3 != 0.0):
   for i in range(1, (numx3)):
      # En lo siguiente se determinar la longitud de espiras en la direccion radial
      Sum3 = Sum3 + (i*dwz3)

   LenghtBob3 = (2.0*numz3*2.0*math.pi*((numx3*L31) + Sum3))

LenghtBob = (LenghtBob1 + LenghtBob2 + LenghtBob3)

print 'La longitud total de alambre necesario para construir las bobinas es (en m):'
print LenghtBob

# Los niveles sirven para graficar todas las lineas de campo, 
# en caso contrario, python grafica las que quiere y comunmente
# son las lineas de mayor magnitud, es decir, no se alcanza a ver
# lo que deseamos
list2 = [numz1,numx1,numz2,numx2,numz3,numx3]
Ilist = [I1, I2, I3]
aux = max(list2)
auxI = max(Ilist)
Iint = ((1.0*B0)/5000.0)
levels = numpy.arange(0.0, (2.0*B0), Iint)

# Gráfica 3D del campo magnético en el plano x-z.
#~ fig = plt.figure()
#~ ax = fig.gca(projection='3d')
#~ surf = ax.plot_surface(X, Z, B1, rstride=1, cstride=1, cmap='coolwarm',
                       #~ linewidth=0, antialiased=False)
#~ ax.set_xlabel('x [cm]',fontsize=20)
#~ ax.set_ylabel('z [cm]',fontsize=20)
#~ ax.set_zlabel('B [G]',fontsize=20)

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el plano y=0')
plt.colorbar(plt.contour(X, Z, B1, levels), shrink=0.8, extend='both')
plt.xlabel('Eje x [cm]')
plt.ylabel('Eje z [cm]')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje z')
plt.plot(apo1,BZtot)
plt.xlabel('z [cm]',fontsize=20)
plt.ylabel('B [G]',fontsize=20)

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje x')
plt.plot(apo1,BXtot)
plt.xlabel('x [cm]',fontsize=20)
plt.ylabel('B [G]',fontsize=20)

plt.show()
