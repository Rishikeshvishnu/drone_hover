import cv2
import numpy as np
import glob
import os
import re

# for sorting the images
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

# per image rmse to filter out images with low rmse
def per_image_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist, cam_name, image_files):
    print(f"\nPer-image RMSE ({cam_name}):")

    rmses = []

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )
        proj = proj.reshape(-1, 2)
        obs = imgpoints[i].reshape(-1, 2)

        err = np.linalg.norm(obs - proj, axis=1)
        rmse = np.sqrt(np.mean(err**2))
        rmses.append(rmse)

        fname = os.path.basename(image_files[i])
        print(f"  {fname}: RMSE = {rmse:.3f} px")

    avg_rmse = np.mean(rmses)
    print(f"Average RMSE ({cam_name}): {avg_rmse:.3f} px")
 

# criteria to stop the corner refinement strategy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3d point assigning to the chessboard corners
objp = np.zeros((6*4,3), np.float32)
objp[:,:2] = np.mgrid[0:6, 0:4].T.reshape(-1,2)

# each square is of length
# objp = objp*34 # or 34
 
# print(objp)

imgpoints_left = [] # for 2d img points in leftcam
imgpoints_right = [] # for 2d img points in rightcam
imgpoints_third = [] # for 2d img points in thirdcam

rightcamimages = sorted(glob.glob('/home/rishikesh/Desktop/drone_hover/camera_calibration/right_cam_imgs_6448/*.jpg'), key = natural_sort_key)
leftcamimages = sorted(glob.glob('/home/rishikesh/Desktop/drone_hover/camera_calibration/left_cam_imgs_6448/*.jpg'), key = natural_sort_key)
thirdcamimages = sorted(glob.glob('/home/rishikesh/Desktop/drone_hover/camera_calibration/third_cam_imgs_6448/*.jpg'), key = natural_sort_key)

objpoints_right = [] # for 3d points

for fname_right in rightcamimages:
    # right camera 
    img_right = cv2.imread(fname_right)
    name_right = os.path.splitext(os.path.basename(fname_right))[0]
    tag_right = name_right[-4:]
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # find chessboard corners for right camera
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6,4), None)

    if ret_right:
        objpoints_right.append(objp.copy())
        
        # right camera refinement
        ref_corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
        imgpoints_right.append(ref_corners_right)
        cv2.drawChessboardCorners(img_right, (6,4), ref_corners_right, ret_right)
        
        # display right images
        cv2.imshow(f"Right-{tag_right}", img_right)
        cv2.waitKey(0)
    else:
        if not ret_right:
            print(f"Chessboard not found in right: {fname_right}")


cv2.destroyAllWindows()

# calibration
retR, camera_matrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints_right, imgpoints_right, gray_right.shape[::-1], None, None, flags = None )
hr, wr = img_right.shape[:2]
refcamera_matrixR, roiR =  cv2.getOptimalNewCameraMatrix(camera_matrixR, distR, (wr,hr), 1, (wr,hr))



objpoints_left = [] # for 3d points


for fname_left in leftcamimages:
    #left camera
    img_left = cv2.imread(fname_left)
    name_left = os.path.splitext(os.path.basename(fname_left))[0]
    tag_left = name_left[-4:]
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    
    # find chessboard corners for left camera
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6,4), None)
    
    # if found in both, add obj points and img points after refining
    if ret_left:
        objpoints_left.append(objp.copy())
        
        # left camera refinement
        ref_corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        imgpoints_left.append(ref_corners_left)
        cv2.drawChessboardCorners(img_left, (6,4), ref_corners_left, ret_left)
        
        # display left images
        cv2.imshow(f"Left-{tag_left}", img_left)
        cv2.waitKey(0)
    else:
        if not ret_left:
            print(f"Chessboard not found in left: {fname_left}")

cv2.destroyAllWindows()

# # calibration
retL, camera_matrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints_left, imgpoints_left, gray_left.shape[::-1], None, None, flags=None)
hl, wl = img_left.shape[:2]
refcamera_matrixL, roiL = cv2.getOptimalNewCameraMatrix(camera_matrixL, distL, (wl,hl), 1, (wl,hl))



# third cam calibration
objpoints_third = [] # for 3d points


for fname_third in thirdcamimages:
    #third camera
    img_third = cv2.imread(fname_third)
    name_third = os.path.splitext(os.path.basename(fname_third))[0]
    tag_third = name_third[-4:]
    gray_third = cv2.cvtColor(img_third, cv2.COLOR_BGR2GRAY)
    
    # find chessboard corners for third camera
    ret_third, corners_third = cv2.findChessboardCorners(gray_third, (6,4), None)
    
    # if found in both, add obj points and img points after refining
    if ret_third:
        objpoints_third.append(objp.copy())
         
        # third camera refinement
        ref_corners_third = cv2.cornerSubPix(gray_third, corners_third, (11,11), (-1,-1), criteria)
        imgpoints_third.append(ref_corners_third)
        cv2.drawChessboardCorners(img_third, (6,4), ref_corners_third, ret_third)
        
        # display third images
        cv2.imshow(f"third-{tag_third}", img_third)
        cv2.waitKey(0)
    else:
        if not ret_third:
            print(f"Chessboard not found in third: {fname_third}")

cv2.destroyAllWindows()

# # calibration
ret3, camera_matrix3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(objpoints_third, imgpoints_third, gray_third.shape[::-1], None, None, flags=None)
hl, wl = img_third.shape[:2]
refcamera_matrix3, roi3 = cv2.getOptimalNewCameraMatrix(camera_matrix3, dist3, (wl,hl), 0, (wl,hl))

     
##### displaying results ####
print("Left Camera Matrix:\n", camera_matrixL)
print("Refined Left Camera Matrix:\n", refcamera_matrixL)

print("\nLeft Distortion Coefficients:\n", distL)
print("\nLeft Reprojection Error:", retL)

print("Right Camera Matrix:\n", camera_matrixR)
print("Refined Right Camera Matrix:\n", refcamera_matrixR)

print("\nRight Distortion Coefficients:\n", distR)
print("\nRight Reprojection Error:", retR)

print("third Camera Matrix:\n", camera_matrix3)
print("Refined third Camera Matrix:\n", refcamera_matrix3)

print("\nthird Distortion Coefficients:\n", dist3)
print("\nthird Reprojection Error:", ret3)

per_image_rmse(
    objpoints_right, imgpoints_right,
    rvecsR, tvecsR,
    camera_matrixR, distR,
    "RIGHT", rightcamimages
)

per_image_rmse(
    objpoints_left, imgpoints_left,
    rvecsL, tvecsL,
    camera_matrixL, distL,
    "LEFT", leftcamimages
)

per_image_rmse(
    objpoints_third, imgpoints_third,
    rvecs3, tvecs3,
    camera_matrix3, dist3,
    "THIRD", thirdcamimages
)

 
store_path = '/home/rishikesh/Desktop/drone_hover/camera_calibration/'

# Save left camera calibration
np.savez(os.path.join(store_path, 'left_camera_calibration0.npz'),
         camera_matrix=camera_matrixL,
         dist_coeffs=distL,
         refined_camera_matrix=refcamera_matrixL,
         roi=roiL,
         reprojection_error=retL)  

# # Save right camera calibration
# np.savez(os.path.join(store_path, 'right_camera_calibration.npz'),
#          camera_matrix=camera_matrixR,
#          dist_coeffs=distR,
#          refined_camera_matrix=refcamera_matrixR,
#          roi=roiR,
#          reprojection_error=retR)    



