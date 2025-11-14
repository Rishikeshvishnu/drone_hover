import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
import json

from cv_bridge import CvBridge
import cv2
import numpy as np


class DroneImageCoordinates(Node):
    def __init__(self):
        super().__init__('drone_image_coordinates')
        
        self.drone_coords_pub = self.create_publisher(String, 'drone_2D_points', 10)
        
        timer_period = 1/10 
        self.create_timer(timer_period, self._timer_callback)

        # start vc , use v4l2 for better cam control
        self.cap1 = cv2.VideoCapture(2, cv2.CAP_V4L2)
        self.cap2 = cv2.VideoCapture(3, cv2.CAP_V4L2)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            self.get_logger().error('Couldnt open one or both webcams. Shutting down')
            return
        
        # set the pixel identifier format, convert the value into an integer code
        self.cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('G','R','B','G'))
        self.cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('G','R','B','G'))

        # set width
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        # set height
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # set fps
        self.cap1.set(cv2.CAP_PROP_FPS, 50)
        self.cap2.set(cv2.CAP_PROP_FPS, 50)
       
    def _display_cams(self):
        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.get_logger().warning('Failed to read frame from one or both webcams.')
                return

            frame1_rs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_rs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            bw1 = cv2.inRange(frame1_rs, 250, 255)
            bw2 = cv2.inRange(frame2_rs, 250, 255)

            conts1, _ = cv2.findContours(bw1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts2, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_with_cnts_1 = frame1.copy()
            frame_with_cnts_2 = frame2.copy()

            centers1 = self._contour_centers(conts1, frame_with_cnts_1)
            centers2 = self._contour_centers(conts2, frame_with_cnts_2)

            midpoint1 = self._midpoint(centers1)
            midpoint2 = self._midpoint(centers2)
            
            points = {'cam1':midpoint1, 'cam2':midpoint2}
            
            
            return points

        #     if midpoint1:
        #         cv2.circle(frame_with_cnts_1, midpoint1, 2, (255, 0, 0), -1)

        #     if midpoint2:
        #         cv2.circle(frame_with_cnts_2, midpoint2, 2, (255, 0, 0), -1)


        #     cv2.imshow('Cam1 feed with m.p', frame_with_cnts_1)
        #     cv2.imshow('Cam2 feed with m.p', frame_with_cnts_2)


        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # cv2.destroyAllWindows()

    def _contour_centers(self, contours, frame):
        centers = []

        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            b, g, r = frame[cy, cx]
            if r > max(b, g): 
                continue
            centers.append((cx, cy))

        centers.sort(key=lambda c: c[0])
        return centers


    def _midpoint(self, centers):

        if len(centers) < 2:
            return None

        (x1, y1), (x2, y2) = centers[:2]
        
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        return {'x':x,'y':y}

    def _timer_callback(self):
        mps = self._display_cams()
        msg = String()
        msg.data = json.dumps(mps)

        self.drone_coords_pub.publish(msg)
    
    
def main():
    rclpy.init()

    img_test = DroneImageCoordinates()

    rclpy.spin(img_test)
    img_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
