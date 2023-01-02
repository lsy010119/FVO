import numpy as np

from geometry_msgs.msg  import Point32


class PoseUpdater:

    def __init__(self, Base, init_pose=np.array([-np.pi/2,0,0,0,0,0])):
        
        self.Base = Base

        self.Base.T_k_w = self.Base.se2SE(init_pose[:3],init_pose[3:])


    def update(self,x):

        T_c_k = self.Base.se2SE(x[-6:-3],x[-3:],inv=True)

        T_c_w = T_c_k @ self.Base.T_k_w

        self.Base.T_c_w = T_c_w

        self.Base.T_c_w_odom.append(T_c_w)

        P_c = Point32(T_c_w[0,3],T_c_w[1,3],T_c_w[2,3])

        self.Base.trajec.points.append(P_c)
        self.Base.trajec_pub.publish(self.Base.trajec)