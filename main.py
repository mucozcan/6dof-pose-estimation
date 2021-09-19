import numpy as np
import cv2

#local imports
import utils


focal_length = 100.0  # for initial instrinsic guess
cx = 960.0
cy = 540.0
dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # no distortion

points_2d = np.load("./points/vr2d.npy")  # (20, 1, 2) 
points_3d = np.load("./points/vr3d.npy")  # (20, 1, 3)

points_2d = np.squeeze(points_2d)  # (20, 2), float32
points_3d = np.squeeze(points_3d)  # (20 ,3), float32

lk_params = dict(winSize  = (21, 21), 
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
poses = []

K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
], dtype=np.float32)  # camera matrix

img1 = cv2.imread("data/img1.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

ret, camera_matrix, dist_coeffs, rotation_vector, translation_vector = cv2.calibrateCamera(
    [points_3d], [points_2d], img1.shape[::-1], K, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS) 


# camera's initial pose
t1 = np.array([
    [0.0],
    [0.0], 
    [0.0]
])

R1 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0 ,0]
])
euler_1 = utils.R_to_euler(R1)
poses.append([t1, euler_1])

# initialize FAST and detect keypoints to track
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
keypoints1 = fast.detect(img1)
keypoints1 = np.array([kp.pt for  kp in keypoints1], dtype=np.float32)

#track keypoints in other frames
#image 2
img2 = cv2.imread("data/img2.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

keypoints1, keypoints2 = utils.track_features(img1, img2, keypoints1, **lk_params)
R2, t2 = utils.get_camera_pose(keypoints1, keypoints2, camera_matrix)

euler_2 = utils.R_to_euler(R2)

poses.append([t2, euler_2])

#image3
img3 = cv2.imread("data/img3.png")
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

keypoints2, keypoints3 = utils.track_features(img2, img3, keypoints2, **lk_params)
R3, t3 = utils.get_camera_pose(keypoints2, keypoints3, camera_matrix)

t3 = t2 + R2.dot(t3)
R3 = R3.dot(R2)
euler_3 = utils.R_to_euler(R3)

poses.append([t3, euler_3])

utils.save_results(poses, save_file="./out/results.txt")

trajectory = np.array([t1, t2, t3], dtype=np.float32)
utils.plot_trajectory(trajectory, save_path="./out/trajectory.png", title="Trajectory of Camera (Bird-Eye View)")