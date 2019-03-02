import numpy as np
import math
import random
import matplotlib.pyplot as plt


from Car import Car
from frame import frame

if __name__ == '__main__':

    t=0.0; dt=0.01

    # Instances
    cadre = frame([0,0],math.pi/120,4,4)
    obj1 = frame([-1.5,0.9],math.pi/3,0.5,0.3)
    obj2 = frame([0.25,-1.0],math.pi/15,0.5,0.5)
    obj3 = frame([1.0,1.1],math.pi/4,0.4,0.4)
    voiture=Car(lag=0.15,ang=50)

    #Objet figure
    Fig=plt.gcf()

    # Demarrer
    voiture.speed = 2.0
    remain_delay=0.0

    poursuivre=1
    t_final = 10.0
    read = 0
    while poursuivre == 1:
        print('Temps: ',round(t,3))
        #Delai restant pour une action
        remain_delay = remain_delay - dt

        #Agir et lire
        if remain_delay<=0.0:


            # Choisir une action
            action = np.random.randint(0,3)#np.random()#########
            remain_delay = 0.1


        #Mettre a jour la position de la voiture
        voiture.move(action,dt)
        read=1

        #Rester dans le cadre
        front= np.array([voiture.capM,voiture.capL,voiture.capR])
        s1 = min(front[:,0])
        s2 = max(front[:,0])
        s3 = min(front[:,1])
        s4 = max(front[:,1])
        if s1<= -cadre.width/2 + voiture.lag or s2 >= cadre.width/2 - voiture.lag or \
                s3<= -cadre.height/2 + voiture.lag or s4>= cadre.height/2 - voiture.lag  :
            voiture.advance(-2*dt)


        #Afficher
        plt.axis([-3,3,-3,3])
        plt.clf()
        plt.ion()
        cadre.show()
        obj1.show()
        obj2.show()
        obj3.show()
        voiture.show()
        plt.axes().set_aspect('equal', 'datalim')
        plt.text(0,-2.75,'Temps: ' +str(round(t,1)))

        # Lire la position
        if read==1:
            [lecture, points]= voiture.read([cadre,obj1,obj2,obj3])
            print(' Action = ',action)
            print(' Lecture: ',lecture)
            ray1=np.array([voiture.capL, np.array(points[0])])
            ray2=np.array([voiture.capM, np.array(points[1])])
            ray3=np.array([voiture.capR, np.array(points[2])])
            plt.plot(ray1[:,0], ray1[:,1], 'k--', linewidth=1.0)
            plt.plot(ray2[:,0], ray2[:,1], 'k--', linewidth=1.0)
            plt.plot(ray3[:,0], ray3[:,1], 'k--', linewidth=1.0)

        plt.draw()
        plt.show()
        plt.pause(0.01)

        #Mettre le temps a jour:
        t=t+dt

        if t>=t_final:
            poursuivre = 0
