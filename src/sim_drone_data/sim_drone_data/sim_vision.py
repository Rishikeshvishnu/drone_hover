import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
import json
import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import qos_profile_sensor_data

from message_filters import Subscriber,ApproximateTimeSynchronizer

class SensorDataProcessing(Node):
    def __init__(self):
        super().__init__('sensor_data_processing')
        self.bridge = CvBridge()

        # enter camera instrinsics
        self.fx = 479.223578
        self.fy = 479.223578
        self.cx = 319.5
        self.cy = 239.5

        
        self.sim_semcamera_data_sub = Subscriber(self, Image,'/camera_left_semantic/labels_map', qos_profile=qos_profile_sensor_data)
        self.sim_dcamera_data_sub = Subscriber(self, Image,'/camera_left/depth_image', qos_profile=qos_profile_sensor_data)
        
        # pair messages whose timestamps are within a tol ( one callback for sem + dep)
        self.sync = ApproximateTimeSynchronizer(
            [self.sim_semcamera_data_sub, self.sim_dcamera_data_sub],
            queue_size=25,
            slop=0.002  
        )

        self.sync.registerCallback(self.manipulate_images)

        # publisher to publish to a topic
        self.sim_camera_data_pub = self.create_publisher(String,'simEKF_camera_data',20)

    def manipulate_images(self, smsg,dmsg):

        labels_rgb = self.bridge.imgmsg_to_cv2(smsg, desired_encoding='rgb8')
        msg = String()
        label_id = labels_rgb[:,:,0]

        mask = (label_id == 255)
        ys, xs = np.where(mask)

        u_f = xs.mean()
        v_f = ys.mean()
        if not (np.isfinite(u_f) and np.isfinite(v_f)):
            self.get_logger().warn("Centroid is non-finite (empty/invalid mask). Skipping frame.")
            return
        
        u, v = int(u_f), int(v_f)
        depth_img = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding='passthrough')

        z = float(depth_img[v,u])
        if (not np.isfinite(z)) or z <= 0.0:
            self.get_logger().warn(f"Invalid depth at (u,v)=({u},{v}): z={z} enc={dmsg.encoding}")
            return
        
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        if not (np.isfinite(x) and np.isfinite(y)):
            self.get_logger().warn(f"Non-finite XYZ computed: x={x}, y={y}, z={z}")
            return
        
        self.get_logger().info(f"XYZ (cam frame): x={x:.3f}, y={y:.3f}, z={z:.3f}")
        point = {'x':x,'y':y,'z':z}

        msg.data = json.dumps(point)

        self.sim_camera_data_pub.publish(msg)

        # vis_img = cv2.cvtColor(labels_rgb, cv2.COLOR_RGB2BGR)
         #if xs.size>0:
        #     cv2.circle(vis_img, (u, v), 3, (0, 255, 0), -1)
        # else:
        #     cv2.putText(vis_img, f"label {255} not found",
        #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #                 (0, 0, 255), 2, cv2.LINE_AA)

        # cv2.imshow("Label map + centroid", vis_img)
        # cv2.waitKey(1)


def main():
    rclpy.init()
    node = SensorDataProcessing()

    rclpy.spin(node)

    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


        
