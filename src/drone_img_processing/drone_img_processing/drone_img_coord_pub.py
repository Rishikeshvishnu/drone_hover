import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
import json

from ultralytics import YOLO

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
    
    ##############################################################################

    def _detect_drone_CNN(self):
        yolo_model = YOLO("/home/rishikesh/Desktop/drone_hover/yolo/yolov8n.pt")

        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            if not ret1 or not ret2:
                self.get_logger().warning('Failed to read frame from one or both webcams.')
                return
            
            frame1_cpy = frame1.copy()
            frame2_cpy = frame2.copy()

            result1 = yolo_model(frame1)[0]
            result2 = yolo_model(frame2)[0]

            if not result1.boxes or not result2.boxes:
                self.get_logger().warning(f'Detection failed: cam1={bool(result1.boxes)} cam2={bool(result2.boxes)}')
            
            box1 = max(result1.boxes, key=lambda b: float(b.conf))
            box2 = max(result2.boxes, key=lambda b: float(b.conf))
            x11, y11, x21, y21 = map(int, box1.xyxy[0])
            x12, y12, x22, y22 = map(int, box2.xyxy[0])

            cv2.rectangle(frame1_cpy, (x11, y11), (x21, y21), (0, 255, 0), 2)
            cv2.rectangle(frame2_cpy, (x12, y12), (x22, y22), (0, 255, 0), 2)

            label1 = f"cfdrone:{float(box1.conf):.2f}"
            label2 = f"cfdrone:{float(box2.conf):.2f}"
            cv2.putText(frame1_cpy, label1, (x11, y11 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame2_cpy, label2, (x12, y12 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('cam1 with yolov8', frame1_cpy)
            cv2.imshow('cam2 with yolov8', frame2_cpy)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
    
    ###############################################################################
    
    def _detect_drone_LEDBlob(self):

        blob_detector = cv2.SimpleBlobDetector_Params()

        blob_detector.filterByColor = True
        blob_detector.blobColor = 255

        # Filter by area
        blob_detector.filterByArea = True
        blob_detector.minArea = 1
        blob_detector.maxArea = 5

        # Filter by circularity (LEDs appear circular)
        blob_detector.filterByCircularity = True
        blob_detector.minCircularity = 0.7

        # Filter by convexity
        #blob_detector.filterByConvexity = True
        #blob_detector.minConvexity = 0.8

        detector = cv2.SimpleBlobDetector_create(blob_detector)


        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            if not ret1 or not ret2:
                self.get_logger().warning('Failed to read frame from one or both webcams.')
                return
            
            frame1_cpy = frame1.copy()
            frame2_cpy = frame2.copy()

            frame1_rs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_rs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            bw1 = cv2.inRange(frame1_rs, 250, 255)
            bw2 = cv2.inRange(frame2_rs, 250, 255)

            kp1 = detector.detect(bw1)
            kp2 = detector.detect(bw2)

            # Draw ALL detected keypoints
            frame1_cpy = cv2.drawKeypoints(frame1_cpy, kp1, None, (0,255,0), 
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            frame2_cpy = cv2.drawKeypoints(frame2_cpy, kp2, None, (0,255,0), 
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            print(f"number of blobs in cam1: {len(kp1)}")
            print(f"number of blobs in cam2: {len(kp2)}")

            cv2.imshow('cam1 with keypoints', frame1_cpy)
            cv2.imshow('cam2 with keypoints', frame2_cpy)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()

    ##############################################################################
       
    def _detect_drone_img_process(self): # needs tweaking
        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.get_logger().warning('Failed to read frame from one or both webcams.')
                return

            frame1_rs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_rs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            bw1 = cv2.inRange(frame1_rs, 245, 255)
            bw2 = cv2.inRange(frame2_rs, 245, 255)

            conts1, _ = cv2.findContours(bw1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts2, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_with_cnts_1 = frame1.copy()
            frame_with_cnts_2 = frame2.copy()

            centers1 = self._contour_centers(conts1, frame_with_cnts_1)
            centers2 = self._contour_centers(conts2, frame_with_cnts_2)

            midpoint1 = self._midpoint(centers1)
            midpoint2 = self._midpoint(centers2)
            
            # points = {'cam1':midpoint1, 'cam2':midpoint2}
            #return points


            if midpoint1:
                x1, y1 = int(midpoint1['x']), int(midpoint1['y'])
                cv2.circle(frame_with_cnts_1, (x1,y1), 1, (0, 255, 0), -1)
            else:
                print("No midpoint on cam 1 for this frame")

            if midpoint2:
                x2, y2 = int(midpoint2['x']), int(midpoint2['y'])
                cv2.circle(frame_with_cnts_2, (x2,y2), 1, (0, 255, 0), -1)
            else:
                print("No midpoint on cam 2 for this frame")
            
            cv2.imshow("Cam1 Midpoint", frame_with_cnts_1)
            cv2.imshow("Cam2 Midpoint", frame_with_cnts_2)
            

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()

    
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

    #############################################################################

    ## best method so far

    def _detect_drone_bgdifference(self):
        bg_cam1 = cv2.imread('/home/rishikesh/Desktop/drone_hover/cam1bg.jpg',cv2.IMREAD_GRAYSCALE)
        bg_cam2 = cv2.imread('/home/rishikesh/Desktop/drone_hover/cam2bg.jpg',cv2.IMREAD_GRAYSCALE)

        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.get_logger().warning('Failed to read frame from one or both webcams.')
                return

            frame1_rs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_rs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            frame_with_centroid1 = frame1.copy()
            frame_with_centroid2 = frame2.copy()

            diff1 = cv2.absdiff(frame1_rs, bg_cam1)
            diff2 = cv2.absdiff(frame2_rs, bg_cam2)

            _,mask1 = cv2.threshold(diff1, 50, 255, cv2.THRESH_BINARY)
            _,mask2 = cv2.threshold(diff2, 50, 255, cv2.THRESH_BINARY)



            # noise removal comes here
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

            cnts1,_ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centroid1 = None
            if cnts1:
                largest = max(cnts1, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centroid1 = [cx, cy]
                    # centroid1 = {'x': cx, 'y': cy}
                    
                    # Draw on frame
                    # cv2.circle(frame_with_centroid1, (cx, cy), 2, (0, 255, 0), -1)


            centroid2 = None
            if cnts2:
                largest = max(cnts2, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centroid2 = [cx, cy]

                    # centroid2 = {'x': cx, 'y': cy}
                    
                    # Draw on frame
                    # cv2.circle(frame_with_centroid2, (cx, cy), 2, (0, 255, 0), -1)

            centroids = {'centroid1': centroid1, 'centroid2':centroid2}

            return centroids

            
        #     cv2.imshow("Cam1 drone Midpoint", frame_with_centroid1)
        #     cv2.imshow("Cam2 drone Midpoint", frame_with_centroid2)

        #     cv2.imshow("Mask 1", mask1)
        #     cv2.imshow("Mask 2", mask2)
            

        #     if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #         break

        # self.cap1.release()
        # self.cap2.release()
        # cv2.destroyAllWindows()


    def _timer_callback(self):
        centroids = self._detect_drone_bgdifference()
        msg = String()
        msg.data = json.dumps(centroids)

        self.drone_coords_pub.publish(msg)
    
    
def main():
    rclpy.init()

    img_test = DroneImageCoordinates()
    # img_test._detect_drone_bgdifference()    

    rclpy.spin(img_test)
    img_test.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
