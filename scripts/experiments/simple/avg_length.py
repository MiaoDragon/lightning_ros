import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
#sys.path.insert(1, '/home/yinglong/Documents/MotionPlanning/baxter/ros_ws/src/lightning_ros/scripts')
from experiments.simple import data_loader_2d, data_loader_r2d, data_loader_r3d
import argparse
import pickle
import time
import os
import numpy as np
import rospy
import sys
# insert at 1, 0 is the script path (or '' in REPL)
from lightning.msg import Float64Array, Float64Array2D
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest, GetMotionPlanResponse
from moveit_msgs.msg import JointConstraint, Constraints
from std_msgs.msg import String, Float64, Int32
# here need to add the path to lightning framework
from run_lightning import Lightning

# multithreading for publishing obstacle information
import threading
"""
to run MPNet planning problem using lightning framework:
1. launch file and set parameters
2. load planning data
3. publish obstacle information to rostopic /obstacles/obc
4. call service of lightning by specifying start and goal
"""

LIGHTNING_SERVICE = "lightning/lightning_get_path"
# shared variable for threading
responded = False
exception = False
def plan(args):
    global responded, exception
    rospy.init_node('lightning_client')
    print('loading...')
    if args.env_type == 's2d':
        data_loader = data_loader_2d
        # create an SE2 state space
        time_limit = 10.
        ratio = 1.
    elif args.env_type == 'c2d':
        data_loader = data_loader_2d
        # create an SE2 state space
        time_limit = 60.
        ratio = 1.
    elif args.env_type == 'r2d':
        data_loader = data_loader_r2d
        # create an SE2 state space
        time_limit = 60.
        ratio = 1.05
    elif args.env_type == 'r3d':
        data_loader = data_loader_r3d
        # create an SE2 state space
        time_limit = 10.
        ratio = 1.

    test_data = data_loader.load_test_dataset(N=args.N, NP=args.NP, s=args.env_idx, sp=args.path_idx, folder=args.data_path)
    obcs, obs, paths, path_lengths = test_data
    obcs = obcs.tolist()
    obs = obs.tolist()
    #paths = paths
    path_lengths = path_lengths.tolist()
    time_env = []
    time_total = []
    fes_env = []   # list of list
    valid_env = []
    total_path_length = 0
    avg_ct = 0
    for i in range(len(path_lengths)):
        for j in range(len(path_lengths[0])):
            if path_lengths[i][j] != 0:
                total_path_length += path_lengths[i][j]
                avg_ct += 1
    print(total_path_length / avg_ct)
parser = argparse.ArgumentParser()
parser.add_argument('--res_path', type=str, default='../lightning_res/')
parser.add_argument('--env_idx', type=int, default=0, help='from which env')
parser.add_argument('--path_idx', type=int, default=0, help='from which path')
parser.add_argument('--N', type=int, default=1)
parser.add_argument('--NP', type=int, default=10)

parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--env_type', type=str, default='s2d')
args = parser.parse_args()
plan(args)
