import cv2
import numpy as np


class KPExtractor:

    def __init__(self,Base):
        
        self.detector = cv2.FastFeatureDetector_create(threshold = Base.th_kpext,nonmaxSuppression=True)


    def detect(self,frame_key):

        kp1 = self.detector.detect(frame_key,None)

        N_detected = len(kp1)

        p_k = np.zeros((N_detected,1,2),dtype=np.float32)

        for i,kp in enumerate(kp1):            
            
            pt1 = kp.pt
            
            self.Base.p_smp = [pt1[0],pt1[1],1]

        return p_k