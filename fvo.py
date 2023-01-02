import rospy
import time

from numpy import array

from lib.base                   import Base,Params
from lib.frame                  import Frame
from lib.init_pose_estimator    import InitPoseEstimator
from lib.pose_estimator         import PoseEstimator
from lib.pose_updater           import PoseUpdater
from lib.keypoint_extractor     import KPExtractor
from lib.keypoint_tracker       import KPTracker
from lib.viz                    import Visualizer

class FVO:

    def __init__(self,Base):
        
        self.Base = Base


    def run(self):

        frame           = Frame(self.Base)
        estimator_init  = InitPoseEstimator(self.Base)
        estimator       = PoseEstimator(self.Base)
        updater         = PoseUpdater(self.Base)
        kpdetector      = KPExtractor(self.Base)
        kptracker       = KPTracker(self.Base)
        viz             = Visualizer(self.Base)


        while not self.Base.is_imgrecieved:

            print("Waiting For Image to be Recieved...",end="\r")
            time.sleep(0.01)


        self.Base.frame_key = self.Base.frame_curr.copy()
        time.sleep(1/self.Base.frame_rate)


        while not rospy.is_shutdown():

            frame_key,frame_curr = self.Base.frame_key,self.Base.frame_curr


            if self.Base.is_keyframe or self.Base.frame_counter == 10:

                self.Base.is_initialized = False
                self.Base.is_keyframe = False
                self.Base.frame_counter = 0

                self.Base.frame_key = frame_curr

                self.Base.T_k_w = self.Base.T_c_w

                self.Base.p_k_smp = kpdetector.detect(frame_key)

                continue


            if not kptracker.track(frame_key, frame_curr, self.Base.p_k_smp):

                self.Base.frame_key = frame_curr
                self.Base.is_keyframe = True

                time.sleep(1/self.Base.frame_rate)

                continue


            p_k,p_c = self.Base.p_k,self.Base.p_c

            if not self.Base.is_initialized:

                w_init,t_init = estimator_init.initial_estimate(p_k,p_c)

                self.Base.x_c[-6:-3] = w_init
                self.Base.x_c[-3:  ] = t_init

                self.Base.x_p[-6:-3] = w_init
                self.Base.x_p[-3:  ] = t_init

                self.Base.is_initialized = True

            else:

                self.Base.x_p = self.Base.x_c.copy()

            self.Base.x_c = estimator.estimate(p_k,p_c,self.Base.x_p)

            # viz.viz_img(frame_key, frame_curr)
            
            updater.update(self.Base.x_c)

            # print(self.Base.x_c[-6:])

            self.Base.frame_counter += 1





                

if __name__ == "__main__":

    # K           = array([[503.791596,0.000000  ,306.540655],
    #                      [0.000000  ,504.248568,240.243212],
    #                      [0.000000  ,0.000000  ,1.000000  ]])

    # dist_coeff  = array([0.077789, -0.156478, -0.000164, 0.001416, 0.000000])
    # size        = (480,640)

    # frame_rate  = 60

    # th_kpext    = 30
    # N           = 100

    # N_iter      = 50
    # stop_crit   = 0.01

    # topic_name  = "/usb_cam/image_raw"


    K           = array([[347.344668, 0.00000000, 317.843671],
                         [0.00000000, 346.900900, 255.698665],
                         [0.00000000, 0.00000000, 1.00000000]])

    dist_coeff  = array([-0.279997, 0.058631, 0.002795, -0.000103, 0.000000])
    size        = (512,640)

    frame_rate  = 60

    th_kpext    = 30
    N           = 100
    N_smp       = 500

    N_iter      = 50
    stop_crit   = 0.01

    topic_name  = "/camera_up/image_raw"

    params = Params(K,dist_coeff,size,\
                    frame_rate,\
                    th_kpext,N,N_smp,\
                    N_iter,stop_crit,\
                    topic_name)

    base = Base(params)

    fvo = FVO(base)

    fvo.run()