import numpy as np
import cv2


class KPTracker:

    def __init__(self,Base):

        self.Base = Base


    def track(self, frame_key, frame_curr, p_k_smp):

        p_c,is_matched,_ = cv2.calcOpticalFlowPyrLK( frame_key, frame_curr, p_k_smp,None)
        
        N = self.Base.N

        try:

            N_tracked = p_c.shape[0]

            if N_tracked < N: 

                print(f"Tracker : Not enough points had been tracked | {N_tracked}/{N}")
                
                return False

            p_c_,p_k_ = p_c[:,-1,:].astype(int),p_k_smp[:,-1,:].astype(int)

            j = 0

            for i in range(N_tracked):

                try:

                    if is_matched[i,0] == 0: 
                        
                        pass

                    else:

                        self.Base.p_c[j,:2] = p_c_[j] 
                        self.Base.p_k[j,:2] = p_k_[j] 
                        j += 1

                        if j == N:

                            return True

                except:

                    pass

            print(f"Tracker : Not enough points had been tracked | {j}/{N}")
            return False

        except: 
            print(f"Tracker : No points had been tracked")
            return False
