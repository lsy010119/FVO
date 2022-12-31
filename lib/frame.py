import numpy as np
import rospy
import cv2

from cv_bridge          import CvBridge
from sensor_msgs.msg    import Image


class Frame:

    def __init__(self,Base):

        self.Base = Base

        rospy.Subscriber(Base.topic_name,Image,self.callback_img,queue_size=1)

        self.K = Base.K
        self.dist_coeff = Base.dist_coeff


    def callback_img(self,msg):

        try:

            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            img_gray_undist = cv2.undistort(img_gray,self.K,self.dist_coeff)

            self.Base.frame_curr = img_gray_undist

            self.Base.is_imgrecieved = True

        except: print("recieve failed")

        