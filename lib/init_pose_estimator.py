import random
import cv2
import numpy as np

class InitPoseEstimator:

    def __init__(self,Base):
        
        self.Base = Base


    def initial_estimate(self,p1,p2):

        E,mask = cv2.findEssentialMat(p2[:,:2].astype(int), p1[:,:2].astype(int), self.Base.K, cv2.RANSAC, 0.999, 1.0)

        _,R,t,_ = cv2.recoverPose(E,p1[:,:2],p2[:,:2],self.Base.K)
        
        w = self.Base.SO2so(R)

        return w,t[:,-1]

    