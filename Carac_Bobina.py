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
	Todos los parametros que se piden a continuacion deben
	ingresarse como flotantes en unidades del SI, i.e. metro,
	Ampere, Tesla. El eje z es definido como el eje de las bobinas
	El punto medio entre las bobinas es tomado como el origen 
	(x,y,z)=(0,0,0)
"""

print 'Introduzca las especificaciones de las bobinas.'
print 'Introduzca estos valores como reales, en unidades del SI.'
print 'ATENCION: Las dimensiones que solicita el programa corresponden'
print 'al punto donde inician las BOBINAS (no el centro de los alambres).'
print ' '

"""
	Se introducen las dimensiones de la bobina de la cual se deseen
	calcular sus propiedades.
"""

# Se pregunta que sistema se desea simular.
T = input('Que sistema se desea desarrollar (Helmholtz=+1, AntHelmholtz=-1) ')
# R : Radio de la bobina (m)
R = input('Cual es el radio mas pequeno de las bobinas (m) ')
# D : Separacion entre los centros de las bobinas (m)
D = input('Cual es la distancia minima entre el punto medio de las bobinas y el centro es estas (m) ')
# numz : El numero de bobinas enrrolladas en z
numz = input('Cuantos alambres tienes en z ')
# numx : El numero de bobinas enrrolladas en x
numx = input('Cuantos alambres tienes en x ')
# dwz : Distancia entre bobinas enrrolladas en z
dwz = input('Cual es la distancia entre las bobinas enrrolladas en z (m) ')
# dwx : Distancia entre bobinas enrrolladas en x
dwx = input('Cual es la distancia entre las bobinas enrrolladas en x (m) ')
# alpha : Pide el gruso del alambre
alpha = input('Cual es el grosor del alambre (m) ')
# I0 : Corriente recorriendo la bobina.
I = input('Cual es la corriente en las bobinas (A) ')

"""
	Constantes físicas.
"""

# u : Constante de permeabilidad del aire
u = 4.0 * math.pi * (10**(-7))
# uB : Magneton de Bohr (J/T)
uB = (9.27400968 * (10**(-24)))
# mLi : Masa del Litio-6 (kg)
mLi = 6.015122795 * (1.66053904 * (10**(-27)))


# Lo siguiente son correcciones para la posición de la bobina
R = (R + (dwx/2.0))
D = (D + (dwz/2.0))

""" 
	Se inicia por definir el espacio donde se graficara el 
    campo magnetico.
"""

list1 = [R, D]
inter = 1.50e-3#max(list1)# Esto permite una grafica mas completa
delta = 0.03003003e-3 # Tamano de pasos en x, y, z, es decir, la resolucion
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
    error = 1e-6
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

def MgLi6(mF , B):     # define la funcion para calcular el valor del momento magnetico para el Litio-6
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

""" Paso siguiente, se definen las matrices para obtener los campos
	y, con estas, se determina la magnitud del campo magnetico
	en cada punto del espacio definido por X, Y, Z
"""

# Bobina superior
# La corriente dada por el usuario, se asumira como la corriente en las 
# bobinas superiores
I1 = I

# Bobina inferior
# La corriente de las bobinas inferiores estara determinada por T
I2 = (I * T)
      
# En las siguientes matrices se llevaran a cabo los calculos para
# obtener el campo magnetico en x, y y z
Bxfin = numpy.zeros(X.shape)
Byfin = Bxfin
Bzfin = Bxfin

# Se calcula el campo magnético en el plano x-z

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de la espira y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de la espira
      R1 = R + (j*dwx)
   
      # Se calcula el campo magnetico de la espira superior
      [Bx,By,Bz] = campos(I,R1,D1,X,Y,Z)
      Bxfin = Bxfin + Bx
      Byfin = Byfin + By
      Bzfin = Bzfin + Bz

      # Se calcula el campo magnetico de la espira inferior
      [Bx,By,Bz] = campos(I2,R1,-D1,X,Y,Z)
      Bxfin = Bxfin + Bx
      Byfin = Byfin + By
      Bzfin = Bzfin + Bz

# Se calcula la magnitud del campo magnetico obtenido anteriormente
B1 = numpy.sqrt((Bxfin**2 + Byfin**2 + Bzfin**2))
# El campo y el espacio se convierte en unidades que nos convengan
B1 = (B1 * 10000) # El campo se graficara en Gauss

# En las siguientes matrices se llevaran a cabo los calculos para
# obtener el campo magnetico en x, y y z
Bxfin = numpy.zeros(X.shape)
Byfin = Bxfin
Bzfin = Bxfin

# Se calcula el campo magnético en el plano x-y

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de la espira
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de la espira
      R1 = R + (j*dwx)
   
      # Se calcula el campo magnetico de la espira superior
      [Bx,By,Bz] = campos(I,R1,D1,X,Z,Y)
      Bxfin = Bxfin + Bx
      Byfin = Byfin + By
      Bzfin = Bzfin + Bz

      # Se calcula el campo magnetico de la espira inferior
      [Bx,By,Bz] = campos(I2,R1,-D1,X,Z,Y)
      Bxfin = Bxfin + Bx
      Byfin = Byfin + By
      Bzfin = Bzfin + Bz

# Se calcula la magnitud del campo magnetico obtenido anteriormente
B2 = numpy.sqrt((Bxfin**2 + Byfin**2 + Bzfin**2))
# El campo y el espacio se convierte en unidades que nos convengan
B2 = (B2 * 10000) # El campo se graficara en Gauss

X = (X * 100) # En cm
Y = (Y * 100) # En cm
Z = (Z * 100) # En cm

# Se calcula el campo magnético en el eje z

BXfin = numpy.zeros(apo1.shape)
BYfin = BXfin
BZfin = BXfin

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de la espira y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de la espira
      R1 = R + (j*dwx)
   
      # Se calcula el campo magnetico de la espira superior
      [BX,BY,BZ] = campos(I,R1,D1,0.0,0.0,apo1)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

      # Se calcula el campo magnetico de la espira inferior
      [BX,BY,BZ] = campos(I2,R1,-D1,0.0,0.0,apo1)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

BZtot = numpy.sqrt((BXfin**2 + BYfin**2 + BZfin**2))

# El campo y el espacio se convierte en unidades que nos convengan
BZtot = (BZtot * 10000) # El campo se graficara en Gauss

# Se calcula el campo magnético en el eje x

BXfin = numpy.zeros(apo1.shape)
BYfin = BXfin
BZfin = BXfin

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de la espira y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de la espira
      R1 = R + (j*dwx)
   
      # Se calcula el campo magnetico de la espira superior
      [BX,BY,BZ] = campos(I,R1,D1,apo1,0.0,0.0)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

      # Se calcula el campo magnetico de la espira inferior
      [BX,BY,BZ] = campos(I2,R1,-D1,apo1,0.0,0.0)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

BXtot = numpy.sqrt((BXfin**2 + BYfin**2 + BZfin**2))

# El campo y el espacio se convierte en unidades que nos convengan
BXtot = (BXtot * 10000) # El campo se graficara en Gauss

apo1 = (apo1 * 100) # En cm

# Se calcula la magnitud del campo magnético en el origen

Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de la espira y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de la espira
      R1 = R + (j*dwx)
   
      # Se calcula el campo magnetico de la espira superior
      [Bx,By,Bz] = campos(I,R1,D1,0.0,0.0,0.0)
      Bx0 = Bx0 + Bx
      By0 = By0 + By
      Bz0 = Bz0 + Bz

      # Se calcula el campo magnetico de la espira inferior
      [Bx,By,Bz] = campos(I2,R1,-D1,0.0,0.0,0.0)
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
    magnetico en cada direccion.
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

# Usando la funcion MgLi6 se pueden calcular los momentos magneticos de cada estado mF:

MgLi61 = MgLi6(-0.5 , B0)  # Para el estado mF = -1/2, ademas depende del valor del campo aplicado
MgLi62 = MgLi6(0.5 , B0) # Para el estado mF = 1/2, ademas depende del valor del campo aplicado

# Lo siguiente es calcular el valor de la frecuencia en cada direccion se tendra:

CR = abs(CurvRho[(pasos/2)+1])
CZ = abs(CurvZ[(pasos/2)+1])
ud = abs(MgLi61)
uu = abs(MgLi62)

w1r = ((1.0/(2.0*math.pi))*((CR*ud*uB/mLi)**(0.5))) # Frecuencia radial en el estado mF = -1/2
w2r = ((1.0/(2.0*math.pi))*((CR*uu*uB/mLi)**(0.5))) # Frecuencia radial en el estado mF = 1/2
w1z = ((1.0/(2.0*math.pi))*((CZ*ud*uB/mLi)**(0.5))) # Frecuencia axial en el estado mF = -1/2
w2z = ((1.0/(2.0*math.pi))*((CZ*uu*uB/mLi)**(0.5))) # Frecuencia axial en el estado mF = 1/2

print 'El valor de la curvatura axial (eje z), es (en G/cm2):'
print CurvZ[(pasos/2)+1]
print 'El valor de la curvatura radial, es (en G/cm2):'
print CurvRho[(pasos/2)+1]
print 'El valor del gradiente axial (eje z), es (en G/cm):'
print (GradZ[(pasos/2)+1])
print 'El valor del gradiente radial, es (en G/cm):'
print (GradRho[(pasos/2)+1])
print 'Frecuencia axial en el estado mF = -1/2 (en Hz):'
print (numpy.sign(CurvZ[(pasos/2)-1])*(-1)*w1z)
print 'Frecuencia axial en el estado mF = 1/2 (en Hz):'
print (numpy.sign(CurvZ[(pasos/2)-1])*(-1)*w2z)
print 'Frecuencia radial en el estado mF = -1/2 (en Hz):'
print (numpy.sign(CurvRho[(pasos/2)-1])*(-1)*w1r)
print 'Frecuencia radial en el estado mF = 1/2 (en Hz):'
print (numpy.sign(CurvRho[(pasos/2)-1])*(-1)*w2r)

""" 
	En lo siguiente se determina el valor de la inductancia para 
    cada una de las bobinas.
"""

# Variables de apoyo para calcular la inductancia
L1 = R
L2 = R + (dwx*numx)
leng = ((numz)*dwz)

# Se determina la inductancia
ap1 = (31.6 * (L1**2) * (numx*numz)**2)
ap2 = (6.0*L1 + 9.0*leng + 10.0*(L2-L1))
Induc = ap1/ap2

# Se indica el valor de la inductancia en pantalla

print 'El valor de la inductancia para cada bobina es (en mH):'
print Induc

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
# ResisCu : Resistividad del cobre
ResisCu = (1.71 * (10**(-8)))

# Se procede a calcular lo solicitado:

ResisBob = (ResisCu * math.pi * (L2 + L1) * (numx * numz))
ResisBob = ResisBob / ((dwx - alpha) * (dwz - alpha))

# Lo anterior correspondio al calculo de la resistividad de la bobina,
# mientras la potencia disipada estara dada por:

PotDisBob = (ResisBob * (I**2))

# Sin embargo, la anterior es la potencia disipada para una sola bobina
# para las dos bobinas se tendra:

PotDisTot = (2.0 * PotDisBob)

print 'La resistencia de una bobina es (en Ohms):'
print ResisBob
print 'El tiempo de descarga de una bobina es (en ms):'
print (5.0*Induc/ResisBob)
print 'La potencia disipada por las bobinas es (en Watts):'
print PotDisTot

# Procedemos a calcular la longitud de alambre necesaria para construir
# esta bobina.

Sum1 = 0.0
for i in range(1, (numx)):
   # En lo siguiente se determinar la longitud de espiras en la direccion radial
   Sum1 = Sum1 + (i*dwz)

LenghtBob = (2.0*numz*2.0*math.pi*((numx*L1) + Sum1))

print 'La longitud total de alambre necesario para construir la bobina es (en m):'
print LenghtBob

# Los niveles sirven para graficar todas las lineas de campo, 
# en caso contrario, python grafica las que quiere y comunmente
# son las lineas de mayor magnitud, es decir, no se alcanza a ver
# lo que deseamos
list2 = [numz,numx]
aux = max(list2)
Iint = ((1.0*B0)/5000.0)
levels = numpy.arange(0.0, (2.0*B0), Iint)

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el plano y=0')
plt.colorbar(plt.contour(X, Z, B1, levels), shrink=0.8, extend='both')
plt.xlabel('Eje x [cm]')
plt.ylabel('Eje z [cm]')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el plano z=0')
plt.colorbar(plt.contour(X, Z, B2, levels), shrink=0.8, extend='both')
plt.xlabel('Eje x [cm]')
plt.ylabel('Eje y [cm]')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje z')
plt.plot(apo1,BZtot)
plt.xlabel('Eje z [cm]')
plt.ylabel('Eje B [G]')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje x')
plt.plot(apo1,BXtot)
plt.xlabel('Eje z [cm]')
plt.ylabel('Eje B [G]')

plt.show()
