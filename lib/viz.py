import numpy as np
import cv2
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self,Base):
        
        self.Base = Base


    def viz_img(self,frame_key,frame_curr):

        # frame_key_ = cv2.cvtColor(frame_key.copy(),cv2.COLOR_GRAY2BGR)
        frame_curr_ = cv2.cvtColor(frame_curr.copy(),cv2.COLOR_GRAY2BGR)

        p1_,p2_ = self.Base.p_k, self.Base.p_c

        for i in range(self.Base.N):

            cv2.circle(frame_curr_, tuple(p1_[i,:2]), 4, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame_curr_, tuple(p2_[i,:2]), 4, (0, 0, 255), 2, cv2.LINE_AA)

            # pt1과 pt2를 이어주는 선 그리기
            cv2.arrowedLine(frame_curr_, tuple(p1_[i,:2]), tuple(p2_[i,:2]), (0, 255, 0), 2)

        cv2.imshow("tracked",frame_curr_)
        cv2.waitKey(1)
