import numpy as np
import cv2

stereo_calib_data = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/stereo_calib.npz')
leftcam_mono_calib = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/left_camera_calibration.npz')
rightcam_mono_calib = np.load('/home/rishikesh/Desktop/drone_hover/camera_calibration/right_camera_calibration.npz')

# stereo calib load
leftcam_K = stereo_calib_data['leftcam_K']
rightcam_K = stereo_calib_data['rightcam_K']
t_mm = stereo_calib_data['t_vec']
R = stereo_calib_data['rot_mat']

# for disparity
fx = leftcam_K[0,0]

# mono calib: only distortion coeffs
dL = leftcam_mono_calib['dist_coeffs']
dR = rightcam_mono_calib['dist_coeffs']

baseline_mm = np.linalg.norm(t_mm)

print("left cam: ",dL)
print("right cam: ",dR)
print("R: ",R)
print("t: ",t_mm)
print("baseline: ",baseline_mm)

#### 
C2_in_C1 = -R.T @ t_mm  # 3x1
print("C2 in C1:", C2_in_C1.ravel())
print("baseline:", np.linalg.norm(C2_in_C1))
####

# # rectification
img_size = (640,480)

R1, R2, P1, P2, Q,roi1, roi2 = cv2.stereoRectify(leftcam_K, dL, rightcam_K, dR, img_size, R, t_mm, flags = cv2.CALIB_ZERO_DISPARITY,newImageSize =(0,0)) #, flags = cv2.CALIB_ZERO_DISPARITY, alpha = 0, newImageSize =(0,0))

# # the below creates a map that will take your raw distorted frame into rectified, undistorted frame (2 per frame(x,y))
mapL_x, mapL_y = cv2.initUndistortRectifyMap(leftcam_K, dL, R1, P1, img_size, m1type = cv2.CV_16SC2)
mapR_x, mapR_y = cv2.initUndistortRectifyMap(rightcam_K, dR, R2, P2, img_size, m1type = cv2.CV_16SC2)

f = P1[0,0]   # (same as P1[1,1] usually)
B = -P2[0,3] / P2[0,0]   # baseline in mm (or whatever unit T used)
print("focal length along x after rect:",f)
print("Recovered baseline:", -P2[0,3] / P2[0,0])
print("Original baseline norm:", np.linalg.norm(t_mm))




