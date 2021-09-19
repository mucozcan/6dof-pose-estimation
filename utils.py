import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def track_features(image_prev, image_cur, kp_prev, **lk_params):
    """Tracks keypoints among the frames.

    Args:
        image_prev (numpy.ndarray): Previous frame.
        image_cur (numpy.ndarray): Current frame.
        kp_prev (numpy.ndarray): Keypoints from previous frame.

    Returns:
        (tuple): Previous keypoints, current keypoints.
    """
    kp_curr, st, err = cv2.calcOpticalFlowPyrLK(image_prev, image_cur, kp_prev, None, **lk_params) 

    st = st.reshape(st.shape[0])

    #to eliminate lost features 
    kp_prev = kp_prev[st == 1] 
    kp_curr = kp_curr[st == 1]

    return kp_prev, kp_curr

def get_camera_pose(kp_prev, kp_curr, camera_matrix):
    """Returns the relative pose of the camera.

    Args:
        kp_prev (numpy.ndarray): Keypoints from previous frame.
        kp_curr (numpy.ndarray): Keypoints from current frame.
        camera_matrix (numpy.ndarray): Camera intrinsics.

    Returns:
        [tuple]: Rotation Matrix, Translation Matrix 
    """
    E, _ = cv2.findEssentialMat(kp_curr, kp_prev, camera_matrix)
    _, R, t, _ = cv2.recoverPose(E, kp_curr, kp_prev, camera_matrix)

    return R, t


def plot_trajectory(trajectory, save_path="./out/trajectory.png", title="Bird-Eye View"): 
    """Plots the trajectory of camera.

    Args:
        trajectory (numpy.ndarray): Estimated camera coordinates.
        save_path (string): Figure save path.
        title (str, optional): Title of plot. Defaults to "Bird-Eye View".
    """
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")
    cdict = {1: 'red', 2: 'blue', 3: 'green'}
    plt.xlim([-0.5, 2.0])
    plt.ylim([-0.5, 2.0])
    plt.plot(trajectory[:, 0], trajectory[:, 2])
    for idx, pt in enumerate(trajectory):
        plt.scatter(pt[0], pt[2], c=cdict[idx+1], label=f"img{idx+1}")
    plt.legend()
    plt.savefig(save_path)
    print(f"Trajectory figure saved to {save_path}")




def R_to_euler(R) : 
    """Calculates euler angles from rotation matrix.
    see the source: https://www.geometrictools.com/Documentation/EulerAngles.pdf

    Args:
        R (numpy.ndarray): Rotation Matrix

    Returns:
        [numpy.ndarray]: Euler angles in X, Y, Z order.
    """
    
    if R[0,2] < 1:

        if R[0,2] > -1:
            x = math.atan2(-R[1,2], R[2,2])
            y = math.asin(R[0,2])
            z = math.atan2(-R[0,1], R[0,0])
        else :
            x = math.atan2(R[1,0], R[1,1])
            y = -(math.pi) / 2
            z = 0
    else:
        x = math.atan2(R[1,0], R[1,1])
        y = (math.pi) / 2
        z = 0
    
    x = math.degrees(x) #roll
    y = math.degrees(y) #pitch
    z = math.degrees(z) #yaw

    return np.array([x, y, z])



def save_results(poses, save_file="./out/results.txt"):
    """ Prints and saves 6DoF poses of camera to a file.
    

    Args:
        poses (numpy.ndarray): Poses of the camera.
        save_file  (str): File save path. Defaults to "./out/results.txt"
    """
    with open(save_file, "w") as res:
        header = "---- 6DoF pose of the camera ----\n"
        res.write(header)
        print(header)
        
        for idx, pose in enumerate(poses):
            t = pose[0]
            euler_angles = pose[1]
            line = f"""
            In img{idx+1}:\n
            X: {t[0][0]}    Y: {t[1][0]}    Z: {t[2][0]}\n
            Roll: {euler_angles[0]}     Pitch: {euler_angles[1]}    Yaw: {euler_angles[2]}\n
            --------------
            """

            res.write(line)
            print(line)

    print(f"Results saved to {save_file}")