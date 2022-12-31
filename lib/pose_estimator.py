import numpy as np
import time
import cv2



class PoseEstimator:

    def __init__(self,Base):
        
        self.Base = Base

        
    def dFdq(self,K,R,q1,q2,P2):
        """
        dp2/dP2

            q2fx   0       -q2(u2 - cx)
            0       q2fy   -q2(v2 - cy)
        
        """
        
        dp2dP2 = - np.array([[q2*K[0,0],0,-(q2**2)*K[0,0]*P2[0]],
                             [0,q2*K[1,1],-(q2**2)*K[1,1]*P2[1]]])

        dP2dP1 = R

        dP1dq1 = np.array([0,0,-1/(q1**2)])

        dFdq = dp2dP2@dP2dP1@dP1dq1

        return dFdq


    def dFdt(self,K,q2,P1,P2):
        """
        dP2dt

            [P]_x | I
        """

        dp2dP2 = - np.array([[q2*K[0,0],0,-(q2**2)*K[0,0]*P2[0]],
                             [0,q2*K[1,1],-(q2**2)*K[1,1]*P2[1]]])


        dP2dt = np.array([[0,P1[2],-P1[1],1,0,0],
                          [-P1[2],0,P1[0],0,1,0],
                          [P1[1],-P1[0],0,0,0,1]])

        dFdt = dp2dP2@dP2dt

        return dFdt


    def schur_comp(self,J,F,H,N):


        H_qq = H[:-6,:-6]
        H_qt = H[:-6,-6:]
        H_tq = H[-6:,:-6]
        H_tt = H[-6:,-6:]
        H_qq_inv = np.diag(np.reciprocal(np.diag(H_qq)))

        b = J.T@F
        b_q = b[:-6]
        b_t = b[-6:]

        del_x = np.zeros((N+6,1))

        del_x[-6:] = np.linalg.inv(-H_tq@H_qq_inv@H_qt + H_tt)@(b_t - H_tq@H_qq_inv@b_q)
        del_x[:-6] = H_qq_inv@(b_q - H_qt@del_x[-6:])

        return del_x


    def cal_delx(self,p1,p2,x,lam):

        K           = self.Base.K
        N           = self.Base.N

        se2SE       = self.Base.se2SE
        proj2D3D    = self.Base.proj2D3D
        proj3D2D    = self.Base.proj3D2D


        dFdq        = self.dFdq
        dFdt        = self.dFdt


        self.Base.F_p = self.Base.F_c.copy()


        for i in range(N):

            T21 = se2SE(x[-6:-3],x[-3:])

            p1_i = p1[i]
            p2_i = p2[i]

            q1_i = x[i]

            P1_i = proj2D3D(p1_i,q1_i)
            P2_i = T21[:3,:3]@P1_i + T21[:3,3]

            q2_i = 1/P2_i[2]

            p2_i_pred = proj3D2D(P2_i)


            dFdq_ = dFdq(K,T21[:3,:3],q1_i,q2_i,P2_i)

            dFdt_ = dFdt(K,q2_i,P1_i,P2_i)


            self.Base.F_c[2*i:2*i+2,0] = p2_i[:2] - p2_i_pred[:2]

            self.Base.J[2*i:2*i+2,i] = dFdq_

            self.Base.J[2*i:2*i+2,N:] = dFdt_


        self.Base.H = self.Base.J.T@self.Base.J

        # self.Base.H_mod = (self.Base.H + lam*np.diag(np.diag(self.Base.H)))
        self.Base.H_mod = (self.Base.H + lam*np.eye(N+6))

        del_x = - self.schur_comp(self.Base.J,self.Base.F_c,self.Base.H_mod,N)

        return del_x


    def estimate(self,p1,p2,x_p):

        lam = 0.01
        stop = self.Base.stop_crit
        N_iter = self.Base.N_iter


        del_x = self.cal_delx(p1,p2,x_p,lam)

        x_1 = x_p
        x_2 = x_1 + del_x[:,0]


        for _ in range(N_iter-1):

            del_x = self.cal_delx(p1,p2,x_2,lam)

            norm_F_1 = np.linalg.norm(self.Base.F_p)
            norm_F_2 = np.linalg.norm(self.Base.F_c)

            if norm_F_2 < norm_F_1:

                if (norm_F_1 - norm_F_2)/norm_F_1 <= stop:
                    
                    print("converged")
                    break

                lam *= 0.5

            else:

                x_1 = x_2
                x_2 = x_1 + del_x[:,0]

                lam *= 1.5

        return x_2