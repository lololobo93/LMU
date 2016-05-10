"""
	La paquetería empleada en este programa se presenta continuación
"""

import math                    # http://docs.python.org/library/math.html
from scipy import special	   # https://docs.scipy.org/doc/scipy/
import numpy                   # numpy.scipy.org/
import matplotlib              # matplotlib.sourceforge.net
# mlab y pyplot son funciones numericas y graficas con estilo de MATLAB
import matplotlib.mlab as mlab # matplotlib.sourceforge.net/api/mlab_api.html
import matplotlib.pyplot as plt# matplotlib.sourceforge.net/api/pyplot_api.html
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.cm as cm

""" 
	Todos los parametros que se piden deben
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

# Se pregunta que sistema se desea simular.
T = input('Que sistema se desea desarrollar (Helmholtz=+1, AntHelmholtz=-1) ')
Lx = input('Cual es la longitud x de la bobina ')
Ly = input('Cual es la longitud y de la bobina ')
# D : Separacion entre los centros de las bobinas.
D = input('Cual es la distancia minima entre el punto medio de las bobinas y el centro de estas ')
# numz : El numero de bobinas enrrolladas en z
numz = input('Cuantos alambres tienes en z ')
# numx : El numero de bobinas enrrolladas en x
numx = input('Cuantos alambres tienes en r ')
# dwz : Distancia entre bobinas enrrolladas en z
dwz = input('Cual es la distancia entre las bobinas enrrolladas en z ')
# dwx : Distancia entre bobinas enrrolladas en x
dwx = input('Cual es la distancia entre las bobinas enrrolladas en r ')
# alpha : Pide el gruso del alambre
alpha = input('Cual es el grosor del alambre ')
# I0 : Corriente recorriendo la bobina.
I = input('Cual es la corriente en las bobinas (positiva) ')
# u : Constante de permeabilidad del aire
u = 4.0 * math.pi * (10**(-7))
# uB : Magneton de Bohr
uB = (9.27400968 * (10**(-24)))
# mLi : Masa del litio-6
mLi = 87.0 * (1.67262158 * (10**(-27)))


# Lo siguiente son correcciones para la posicion de la bobina
D = (D + (dwz/2.0))
Lx = (Lx + (dwx))
Ly = (Ly + (dwx))

""" Se inicia por definir el espacio donde se graficara el 
    campo magnetico.
"""

list1 = [Lx,Ly,D]
inter = (0.5)*max(list1)# Esto permite una grafica mas completa
delta = 0.1e-2 # Tamano de pasos en x, y, z, es decir, la resolucion
apo1 = numpy.arange(-inter, inter, delta) # crea los pasos
apo2 = numpy.arange(-inter, inter, delta) # crea los pasos
X, Z = numpy.meshgrid(apo1, apo2) 
Y = 0.0

""" A continuacion se escribiran las ecuaciones del campo magnetico
	para una sola bobina, basados en las ecuaciones (1) y (2)
	de la referencia mencionada al comienzo de este trabajo.
	El campo magnetico se obtendra en coordenadas cilindricas (rho,phi,z),
	posteriormente se transcriben a cartesianas.
"""

def BarX(I,D,L1,L2,x,y,z):
	M = 4.0 * math.pi * 0.0000001
	error = 1e-6
	
	con1 = M*I/(4.0*math.pi)
	con2 = 1.0/((y-L2/2.0)**2+(z-D)**2)
	con3 = (L1-2.0*x)
	con4 = (L1+2.0*x)
	con5 = (4.0*((z-D)**2+(y-L2/2.0)**2)+(L1-2.0*x)**2)**0.5
	con6 = (4.0*((z-D)**2+(y-L2/2.0)**2)+(L1+2.0*x)**2)**0.5
	
	bas1 = con3/con5
	bas2 = con4/con6
	bas3 = bas1 + bas2
	bas4 = bas3*con2*con1
	
	BX = 0.0
	BY = -(z-D)*bas4
	BZ = (y-L2/2.0)*bas4
	
	return [BX,BY,BZ]
	
def BarY(I,D,L1,L2,x,y,z):
	M = 4.0 * math.pi * 0.0000001
	error = 1e-6
	
	con1 = M*I/(4.0*math.pi)
	con2 = 1.0/((x-L1/2.0)**2+(z-D)**2)
	con3 = (L2-2.0*y)
	con4 = (L2+2.0*y)
	con5 = (4.0*((z-D)**2+(x-L1/2.0)**2)+(L2-2.0*y)**2)**0.5
	con6 = (4.0*((z-D)**2+(x-L1/2.0)**2)+(L2+2.0*y)**2)**0.5
	
	bas1 = con3/con5
	bas2 = con4/con6
	bas3 = bas1 + bas2
	bas4 = bas3*con2*con1
	
	BX = (z-D)*bas4
	BY = 0.0
	BZ = -(x-L1/2.0)*bas4
	
	return [BX,BY,BZ]

def BSquare(I,D,L1,L2,x,y,z):     # define funcion get_bz
    # Lo primero es anotar las funciones de importancia
    
    BX1X = BarX(-I,D,L1,-L2,x,y,z)
    BX2X = BarX(I,D,L1,L2,x,y,z)
    
    BY1Y = BarY(I,D,-L1,L2,x,y,z)
    BY2Y = BarY(-I,D,L1,L2,x,y,z)
    
    BST = []
    
    for i in range(0,3):
		BST.append(BX1X[i]+BX2X[i]+BY2Y[i]+BY1Y[i])
    
    # Finalmente se devuelven los resultados de la funcion
    return BST

""" Paso siguiente, se definen las matrices para obtener los campos
	y, con estas, se determina la magnitud del campo magnetico
	en cada punto del espacio definido por X, Y, Z
"""

# Bobina superior
# La corriente dada por el usuario, se asumira como la corriente en las 
# bobinas superiores
I1 = I

# Bobinta inferior
# La corriente de las bobinas inferiores estara determinada por T
I2 = (I * T)
      
# En las siguientes matrices se llevaran a cabo los calculos para
# obtener el campo magnetico en x, y y z
Bxfin = numpy.zeros(X.shape)
Byfin = Bxfin
Bzfin = Bxfin

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de las bobinas y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de las bobinas
      Lx1 = Lx + (2.0*j*dwx)
      Ly1 = Ly + (2.0*j*dwx)
   
      # Se calcula el campo magnetico de la bobina superior
      [Bx,By,Bz] = BSquare(I1,D1,Lx1,Ly1,X,Y,Z)
      Bxfin = Bxfin + Bx
      Byfin = Byfin + By
      Bzfin = Bzfin + Bz

      # Se calcula el campo magnetico de la bobina inferior y se suma
      # con el campo obtenido para la bobina superior
      [Bx,By,Bz] = BSquare(I2,-D1,Lx1,Ly1,X,Y,Z)
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

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de las bobinas y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de las bobinas
      Lx1 = Lx + (2.0*j*dwx)
      Ly1 = Ly + (2.0*j*dwx)
   
      # Se calcula el campo magnetico de la bobina superior
      [Bx,By,Bz] = BSquare(I1,D1,Lx1,Ly1,X,Z,Y)
      Bxfin = Bxfin + Bx
      Byfin = Byfin + By
      Bzfin = Bzfin + Bz

      # Se calcula el campo magnetico de la bobina inferior y se suma
      # con el campo obtenido para la bobina superior
      [Bx,By,Bz] = BSquare(I2,-D1,Lx1,Ly1,X,Z,Y)
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

BXfin = numpy.zeros(apo1.shape)
BYfin = BXfin
BZfin = BXfin

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de las bobinas y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de las bobinas
      Lx1 = Lx + (2.0*j*dwx)
      Ly1 = Ly + (2.0*j*dwx)
   
      # Se calcula el campo magnetico de la bobina superior
      [BX,BY,BZ] = BSquare(I1,D1,Lx1,Ly1,0.0,0.0,apo1)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

      # Se calcula el campo magnetico de la bobina inferior y se suma
      # con el campo obtenido para la bobina superior
      [BX,BY,BZ] = BSquare(I2,-D1,Lx1,Ly1,0.0,0.0,apo1)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

BZtot = numpy.sqrt((BXfin**2 + BYfin**2 + BZfin**2))

# El campo y el espacio se convierte en unidades que nos convengan
BZtot = (BZtot * 10000) # El campo se graficara en Gauss

BXfin = numpy.zeros(apo1.shape)
BYfin = BXfin
BZfin = BXfin

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de las bobinas y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de las bobinas
      Lx1 = Lx + (2.0*j*dwx)
      Ly1 = Ly + (2.0*j*dwx)
   
      # Se calcula el campo magnetico de la bobina superior
      [BX,BY,BZ] = BSquare(I1,D1,Lx1,Ly1,apo1,0.0,0.0)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

      # Se calcula el campo magnetico de la bobina inferior y se suma
      # con el campo obtenido para la bobina superior
      [BX,BY,BZ] = BSquare(I2,-D1,Lx1,Ly1,apo1,0.0,0.0)
      BXfin = BXfin + BX
      BYfin = BYfin + BY
      BZfin = BZfin + BZ

BXtot = numpy.sqrt((BXfin**2 + BYfin**2 + BZfin**2))

# El campo y el espacio se convierte en unidades que nos convengan
BXtot = (BXtot * 10000) # El campo se graficara en Gauss

apo1 = (apo1 * 100) # En cm

Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0

for i in range(0, (numz)):
   # La siguiente cantidad corresponde a la distancia entre el centro
   # de las bobinas y el origen
   D1 = D + (i*dwz)
    
   for j in range(0, (numx)):
      # La siguiente cantidad corresponde al radio de las bobinas
      Lx1 = Lx + (2.0*j*dwx)
      Ly1 = Ly + (2.0*j*dwx)
   
      # Se calcula el campo magnetico de la bobina superior
      [Bx,By,Bz] = BSquare(I1,D1,Lx1,Ly1,0.0,0.0,0.0)
      Bx0 = Bx0 + Bx
      By0 = By0 + By
      Bz0 = Bz0 + Bz

      # Se calcula el campo magnetico de la bobina inferior y se suma
      # con el campo obtenido para la bobina superior
      [Bx,By,Bz] = BSquare(I2,-D1,Lx1,Ly1,0.0,0.0,0.0)
      Bx0 = Bx0 + Bx
      By0 = By0 + By
      Bz0 = Bz0 + Bz

B0 = numpy.sqrt((Bx0**2 + By0**2 + Bz0**2))

# El campo y el espacio se convierte en unidades que nos convengan
B0 = (B0 * 10000) # El campo se graficara en Gauss

print ' '
print 'El campo magnetico en el centro tiene una magnitud (en Gauss):'
print  B0

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
print abs(GradZ[(pasos/2)-2])
print 'El valor del gradiente radial, es (en G/cm):'
print abs(GradRho[(pasos/2)-2])

""" En lo siguiente se determina el valor de la inductancia para 
    cada una de las bobinas.
"""

# Los niveles sirven para graficar todas las lineas de campo, 
# en caso contrario, python grafica las que quiere y comunmente
# son las lineas de mayor magnitud, es decir, no se alcanza a ver
# lo que deseamos
list2 = [numz,numx]
aux = max(list2)
Iint = ((1.0 * aux * I) / 200.0)
levels = numpy.arange(0.0, (1.0 * aux * I), Iint)

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
plt.xlabel('Eje x (cm)')
plt.ylabel('Eje z (cm)')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el plano z=0')
plt.colorbar(plt.contour(X, Z, B2, levels), shrink=0.8, extend='both')
plt.xlabel('Eje x (cm)')
plt.ylabel('Eje y (cm)')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje z')
plt.plot(apo1,BZtot)
plt.xlabel('Eje z (cm)')
plt.ylabel('Eje B (Gauss)')

plt.figure()
plt.title('Magnitud del campo magnetico (Gauss) en el eje x')
plt.plot(apo1,BXtot)
plt.xlabel('Eje x (cm)')
plt.ylabel('Eje B (Gauss)')

plt.show()
