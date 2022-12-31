import rospy
import matplotlib.pyplot as plt

from numpy              import array,zeros,ones,eye,trace,cos,sin,arccos,pi
from numpy.linalg       import norm

from cv_bridge          import CvBridge
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point32
from sensor_msgs.msg    import PointCloud


class Keypoint:

    def __init__(self, id=-1, coord2D=array([]),coord3D=array([])):
        
        self.id = id
        self.coord2D = coord2D
        self.coord3D = coord3D

        self.status = 1
        self.age = 0


class Params:

    def __init__(self,K,dist_coeff,size,\
                      frame_rate,\
                      th_kpext,N,\
                      N_iter,stop_crit,\
                      topic_name):
        
        self.K = K
        self.dist_coeff = dist_coeff
        self.size = size

        self.frame_rate = frame_rate

        self.th_kpext = th_kpext
        self.N = N

        self.N_iter = N_iter
        self.stop_crit = stop_crit

        self.topic_name = topic_name


class Base:


    def __init__(self, params):

        ''' Params '''

        self.K = params.K

        self.fx,self.fy,self.cx,self.cy = self.K[0,0],self.K[1,1],self.K[0,2],self.K[1,2]

        self.K_inv = array([ [1/self.fx, 0,         -self.cx/self.fx],
                             [0,         1/self.fy, -self.cy/self.fy],
                             [0,         0,         1               ]])

        self.dist_coeff = params.dist_coeff

        self.height,self.width = params.size

        ''' Frames '''

        self.frame_curr = zeros(params.size,        dtype=int)
        self.frame_key  = zeros(params.size,        dtype=int)

        self.frame_rate = params.frame_rate

        self.frame_counter = 0

        ''' Keypoints '''

        self.N = params.N

        self.th_kpext = params.th_kpext

        self.p_k_smp    = 0

        self.p_k        = ones((self.N,3),          dtype=int)
        self.p_c        = ones((self.N,3),          dtype=int)

        ''' Optimization '''

        self.F_p        = zeros((self.N*2,1),       dtype=float)
        self.F_c        = zeros((self.N*2,1),       dtype=float)
        
        self.J          = zeros((self.N*2,self.N+6),dtype=float)
        
        self.H          = zeros((self.N+6,self.N+6),dtype=float)
        self.H_mod      = zeros((self.N+6,self.N+6),dtype=float)
        
        self.x_p        = 0.1*ones((self.N+6),    dtype=float)
        self.x_c        = 0.1*ones((self.N+6),    dtype=float)

        self.N_iter = params.N_iter

        self.stop_crit = params.stop_crit

        ''' Odometry '''

        self.T_k1w      = zeros((4,4),              dtype=float)
        self.T_k2k1     = zeros((4,4),              dtype=float)

        ''' Flags '''

        self.is_imgrecieved = False
        self.is_initialized = False
        self.is_keyframe = True

        ''' ROS '''

        rospy.init_node("FVO")

        self.topic_name = params.topic_name

        self.trajec = PointCloud()
        self.trajec.header.frame_id = "map"       
        self.trajec_pub = rospy.Publisher("/traj",PointCloud,queue_size=1)


    ''' Lie Algebra '''

    def SO2so(self,R):

        theta = arccos((trace(R)-1)/2)
        
        if theta == 0 or theta == pi:

            w = array([0,0,0])

        else:

            nw_hat = (1/(2*sin(theta)))*(R-R.T)

            w = array([nw_hat[2,1],nw_hat[0,2],nw_hat[1,0]])*theta

        return w


    def so2SO(self,w):

        angle = norm(w)

        if angle == 0:  
            
            R = eye(3)
        
        else:

            n_w = w/angle

            n_w_hat = array([[   0, -n_w[2],  n_w[1]],
                             [ n_w[2],    0, -n_w[0]],
                             [-n_w[1],  n_w[0],    0]])

            R = (1-cos(angle))*(n_w_hat @ n_w_hat) + sin(angle)*n_w_hat + eye(3)

        return R


    def se2SE(self,w,t,inv=False):

        if inv:

            T = eye(4)

            angle = norm(w)

            if angle == 0:  
                
                pass
            
            else:

                n_w = w/angle

                n_w_hat = array([[   0, -n_w[2],  n_w[1]],
                                 [ n_w[2],    0, -n_w[0]],
                                 [-n_w[1],  n_w[0],    0]])

                R = (1-cos(angle))*(n_w_hat @ n_w_hat) + sin(angle)*n_w_hat + eye(3)
            
            T[:3,:3] = R.T

            T[:3,3] = -R.T@t

            return T


        else:

            T = eye(4)

            angle = norm(w)

            if angle == 0:  
                
                pass
            
            else:

                n_w = w/angle

                n_w_hat = array([[   0, -n_w[2],  n_w[1]],
                                 [ n_w[2],    0, -n_w[0]],
                                 [-n_w[1],  n_w[0],    0]])

                T[:3,:3] = (1-cos(angle))*(n_w_hat @ n_w_hat) + sin(angle)*n_w_hat + eye(3)

            T[:3,3] = t

            return T


    ''' Projection '''

    def proj3D2D(self,P):

        K = self.K

        q = 1/P[2]

        p = q*K@P

        return p


    def proj2D3D(self,p,q):

        K_inv = self.K_inv

        P = (1/q)*K_inv@p

        return P

    
    ''' Warp '''

    def proj_warp(self,img,dst,T):

        h,w = img.shape

        proj2D3D = self.proj2D3D
        proj3D2D = self.proj3D2D

        for u in range(w):
            for v in range(h):

                intensity = img[v,u]

                p1 = array([u,v,1])
                P1 = proj2D3D(p1,1)

                P2 = T[:3,:3]@P1 + T[:3,3]
                p2 = proj3D2D(P2)

                try: dst[p2[1],p2[0]] = intensity
                
                except: pass

        return dst



