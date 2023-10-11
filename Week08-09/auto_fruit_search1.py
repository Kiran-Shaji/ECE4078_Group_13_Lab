# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import pygame

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
from operate import Operate



def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    desired_angle = np.arctan2((waypoint[1]-robot_pose[1]),(waypoint[0]-robot_pose[0])) 

    print("Desired angle: {}".format(desired_angle))

    lv = 0
    rv = 0

    wheel_vel = 30 # tick
    
    # turn towards the waypoint
    full_turn_time = (baseline*2*np.pi)/(2*scale*wheel_vel) # time to do 1 rotation, so we need to rotate to get to the desired angle
    in_rad = robot_pose[2]
    while in_rad > 2*np.pi:
        in_rad -= 2*np.pi

    while in_rad < -2*np.pi:
        in_rad += 2*np.pi
    
    # we only need to do a partial turn. I
    angle_delta = desired_angle - in_rad
    if angle_delta > np.pi:
        angle_delta -= 2*np.pi
    elif angle_delta < -np.pi:
        angle_delta += 2*np.pi
    
    turn_time = full_turn_time*(angle_delta)/(2*np.pi)
    #print(turn_time[0])
    if turn_time < 0: # if the robot is turning the wrong way just swap the 1 and -1.
        turn_direction = -1
    else:
        turn_direction = 1
    turn_time = abs(turn_time)

    #
    print("Turning for {:.2f} seconds".format(turn_time))
    if turn_time > 0.0:
        lv, rv = operate.pibot.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=turn_time)
    #print([lv, rv])
    drive_meas = measure.Drive(lv, -rv, turn_time)
    time.sleep(0.2)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.draw(canvas)
    pygame.display.update()
    for _ in range(3):
        
        lv, rv = operate.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)
        operate.draw(canvas)
        pygame.display.update()

    
    # after turning, drive straight to the waypoint
    distance = np.sqrt((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)
    drive_time = distance/(scale*wheel_vel)
    print("Driving for {:.2f} seconds".format(drive_time))
    lv, rv = operate.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    drive_meas = measure.Drive(lv, -rv, drive_time)
    time.sleep(0.2)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.draw(canvas)
    pygame.display.update()
    for _ in range(3):
        
        lv, rv = operate.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)
        operate.draw(canvas)
        pygame.display.update()


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = operate.ekf.robot.state.squeeze().tolist() # replace with your calculation
    print("____________________")
    while robot_pose[2] > 2*np.pi:
        robot_pose[2]  -= 2*np.pi

    while robot_pose[2]  < -2*np.pi:
        robot_pose[2]  += 2*np.pi
    print(robot_pose)
    print("____________________")
    ####################################################

    return robot_pose

# main loop
if __name__ == "__main__":
      
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3

    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    if True:
        operate = Operate(args)
        print("OPERATAE INIT")



    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    waypoint_list = [[0.3, -0.3], [0.8, -0.31],[0,0] , [-0.1, 0.5], [0,0]]


    # operate.ekf.taglist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # this for loop initialises the covariance of the 10 markers that we just added to be super low
    # for lm in range(len(aruco_true_pos)):
                        
            
    #         lm_inertial = aruco_true_pos[lm].reshape(-1, 1)

            
    #         operate.ekf.markers = np.concatenate((operate.ekf.markers, lm_inertial), axis=1)

    #         # Create a simple, large covariance to be fixed by the update step
    #         operate.ekf.P = np.concatenate((operate.ekf.P, np.zeros((2, operate.ekf.P.shape[1]))), axis=0)
    #         operate.ekf.P = np.concatenate((operate.ekf.P, np.zeros((operate.ekf.P.shape[0], 2))), axis=1)
    #         operate.ekf.P[-2,-2] = 1e-20
    #         operate.ekf.P[-1,-1] = 1e-20

    lms = []
    for i,lm in enumerate(aruco_true_pos):
        measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1, covariance=(0.0*np.eye(2)))
        lms.append(measure_lm)
    operate.ekf.add_fixed_landmark(lms)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    operate.ekf_on = True


    while start:
        operate.update_keyboard()
        operate.take_pic()
        


        lv, rv = operate.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)
        
       
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        robot_pose = get_robot_pose()
        drive_to_point(waypoint_list[0], robot_pose)
        robot_pose = get_robot_pose()
        
        if len(waypoint_list) > 0:
            waypoint_list.pop(0)
        if len(waypoint_list) == 1:
            start == False



    

    # The following is only a skeleton code for semi-auto navigation
    while False:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
    