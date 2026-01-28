import numpy as np
import cv2
from scipy.optimize import least_squares
import os

# 640, 480

def triangulate_points(R, t, K1, K2, pts1, pts2):

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K2 @ np.hstack((R, t.reshape(3,1)))

    # OpenCV expects float32 and shape (2,N)
    pts1 = pts1.astype(np.float32).T   # (2, N)
    pts2 = pts2.astype(np.float32).T   # (2, N)

    # Triangulate
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert from homogeneous
    pts3d = (pts4d[:3] / pts4d[3]).T  # Nx3

    # Check if points are in front of both cameras
    z1 = pts3d[:, 2] > 0
    pts_cam2 = (R @ pts3d.T + t.reshape(3,1)).T
    z2 = pts_cam2[:, 2] > 0

    return int(np.sum(z1 & z2)), pts3d

# loading the 2d-2d corres from both cam imgs
data = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/clickpoints_stereo.npz')

left_cam_points = data["left_cam_clickpoints_cam1"]
right_cam_points = data["right_cam_clickpoints_cam2"]

# loading cam calib matrices
left_mono_calib_data = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/left_camera_calibration.npz')
right_mono_calib_data = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/right_camera_calibration.npz')

Kl = left_mono_calib_data['camera_matrix']
Kr = right_mono_calib_data['camera_matrix']

# going from arbitrary scale to mm

metric_data = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/dreal.npz')
rightcam_pt = metric_data['pt1']
leftcam_pt = metric_data['pt2']


# computing fundamental matrix 
F, mask = cv2.findFundamentalMat(left_cam_points, right_cam_points, cv2.USAC_MAGSAC)

inliers1 = left_cam_points[mask.ravel()==1]
inliers2 = right_cam_points[mask.ravel()==1]

print("Fundamental matrix: \n",F)
print("Total Number of points: ",len(left_cam_points))
print("Number of inlier points: ",int(mask.sum()))

# computing essential matrix

E = Kr.T @ F @ Kl
print("Essential matrix: \n",E)


_,R,t,_ = cv2.recoverPose(E, inliers1, inliers2, Kl)

print("Rotation matrix(R->L): \n",R)
print("translation vector: \n",t)
print("direction of translation vector: \n",t/np.linalg.norm(t))


_, pts3d = triangulate_points(R, t, Kl, Kr, leftcam_pt, rightcam_pt)

# measured distance
dreal = 600 # in mm

print("pts3d: ",pts3d)

dcalc = np.linalg.norm(pts3d[1] - pts3d[0])
scale = dreal / dcalc
print("Scale factor =", scale)

t_mm = scale * t

print("t in mm =", t_mm)
print("baseline length in mm =", np.linalg.norm(t_mm))

store_path = '/home/rishikesh/Desktop/drone_hover/camera_calibration'
full_store_path = os.path.join(store_path,"stereo_calib.npz")

np.savez(full_store_path,rot_mat = R, t_vec = t_mm,scale = 1, leftcam_K = Kl, rightcam_K = Kr)

