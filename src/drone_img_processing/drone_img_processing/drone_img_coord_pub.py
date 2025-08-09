import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
import numpy as np

from ultralytics import YOLO


class DroneImageCoordinates(Node):
    def __init__(self):
        super().__init__('drone_image_coordinates')
        
        self.drone_coords_pub = self.create_publisher(Image, 'drone_image', 10)
        
        timer_period = 1/10 # 10 hertz
        self.create_timer(timer_period, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Couldnt open webcam. Shutting down')
            return
        
        self.bridge = CvBridge()

        self.model = YOLO('yolo/runs/detect/train3/weights/best.pt')
        self.model.to('cuda:0')
        
    def drone_yolo(self):
        while True:
            ret,frame = self.cap.read()
            if not ret:
                self.get_logger().warning('Failed to read frame from webcam.')
                return
            
            cv_img = frame
            bb_img = cv_img.copy()

            res = self.model(cv_img, device = 'cuda:0')[0]

            if len(res.boxes.data) == 0:
                self.get_logger().warning('Failed to read detect drone')
            else:
                detection = res.boxes.data.tolist()
                best_detect = max(detection, key = lambda det: det[4])

                x1,y1,x2,y2,conf,cls = best_detect
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                cv2.rectangle(bb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                cv2.circle(bb_img, (cx,cy), radius=5, color=(0, 0, 255), thickness=-1)

                #bb.center.x = (x1 + x2)/2
                #bb.center.y = (y1 + y2)/2
                
            cv2.imshow('YOLO live feed', bb_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # def drone_cp(self):
    #     while True:
    #         ret,frame = self.cap.read()
    #         if not ret:
    #             self.get_logger().warning('Failed to read frame from webcam.')
    #             return
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #         # blue color range
    #         lower_blue = (100, 150, 0)
    #         upper_blue = (140, 255, 255)

    #         mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #         kernel = np.ones((5, 5), np.uint8)
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #         cv2.imshow('Mask', mask)

    #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         led_centers = []
    #         for contour in contours:
    #             M = cv2.moments(contour)
    #             if M["m00"] != 0:
    #                 cx = int(M["m10"] / M["m00"])
    #                 cy = int(M["m01"] / M["m00"])
    #                 led_centers.append((cx, cy))
            
    #         result = frame.copy()

    #         if len(led_centers) == 2:
    #             point1 = led_centers[0]
    #             point2 = led_centers[1]
    #             midpoint_x = (point1[0] + point2[0]) // 2
    #             midpoint_y = (point1[1] + point2[1]) // 2
    #             midpoint = (midpoint_x, midpoint_y)
    #             cv2.circle(result, midpoint, 5, (0, 0, 255), -1)

    #         else:
    #             self.get_logger().info('Did not detect exactly two blue LEDs.')

            
    #         cv2.imshow('Result', result)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     cv2.destroyAllWindows()

        
    
    def timer_callback(self):
        ret,frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')
            self.drone_coords_pub.publish(msg)
        else:
            self.get_logger().warning('Failed to read frame from webcam.')
    
    
def main():
    rclpy.init()

    img_test = DroneImageCoordinates()

    drone_cp_test = img_test.drone_yolo()

    # rclpy.spin(img_test)
    img_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
