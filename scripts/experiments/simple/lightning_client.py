import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
#sys.path.insert(1, '/home/yinglong/Documents/MotionPlanning/baxter/ros_ws/src/lightning_ros/scripts')
from experiments.simple import data_loader_2d, data_loader_r2d, data_loader_r3d
from experiments.simple import plan_s2d, plan_c2d, plan_r2d, plan_r3d
from tools import plan_general
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
        IsInCollision = plan_s2d.IsInCollision
        data_loader = data_loader_2d
        # create an SE2 state space
        time_limit = 15.
        ratio = 1.
    elif args.env_type == 'c2d':
        IsInCollision = plan_c2d.IsInCollision
        data_loader = data_loader_2d
        # create an SE2 state space
        time_limit = 60.
        ratio = 1.
    elif args.env_type == 'r2d':
        IsInCollision = plan_r2d.IsInCollision
        data_loader = data_loader_r2d
        # create an SE2 state space
        time_limit = 60.
        ratio = 1.05
    elif args.env_type == 'r3d':
        IsInCollision = plan_r3d.IsInCollision
        data_loader = data_loader_r3d
        # create an SE2 state space
        time_limit = 15.
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
    # setup publisher
    obc_pub = rospy.Publisher('lightning/obstacles/obc', Float64Array2D, queue_size=10)
    obs_pub = rospy.Publisher('lightning/obstacles/obs', Float64Array, queue_size=10)
    obs_i_pub = rospy.Publisher('lightning/obstacles/obs_i', Int32, queue_size=10)
    length_pub = rospy.Publisher('lightning/planning/path_length_threshold', Float64, queue_size=10)

    for i in xrange(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        obc = obcs[i]
        # publishing to ROS topic
        obc_msg = Float64Array2D([Float64Array(obci) for obci in obc])
        obs_msg = Float64Array(obs[i])
        obs_i_msg = Int32(i)
        for j in xrange(len(paths[0])):
            # check if the start and goal are in collision
            # if so, then continue
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            elif IsInCollision(paths[i][j][0], obc) or IsInCollision(paths[i][j][path_lengths[i][j]-1], obc):
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            else:
                valid_path.append(1)
                # obtaining the length threshold for planning
                data_length = 0.
                for k in xrange(path_lengths[i][j]-1):
                    data_length += np.linalg.norm(paths[i][j][k+1]-paths[i][j][k])
                length_msg = Float64(data_length * ratio)
                # call lightning service
                request = GetMotionPlanRequest()
                request.motion_plan_request.group_name = 'base'
                for k in xrange(len(paths[i][j][0])):
                    request.motion_plan_request.start_state.joint_state.name.append('%d' % (k))
                    request.motion_plan_request.start_state.joint_state.position.append(paths[i][j][0][k])

                request.motion_plan_request.goal_constraints.append(Constraints())
                for k in xrange(len(paths[i][j][0])):
                    request.motion_plan_request.goal_constraints[0].joint_constraints.append(JointConstraint())
                    request.motion_plan_request.goal_constraints[0].joint_constraints[k].position = paths[i][j][path_lengths[i][j]-1][k]
                request.motion_plan_request.allowed_planning_time = time_limit

                responded = False
                exception = False
                def publisher():
                    global responded, exception
                    while not responded and not exception:
                        print('sending obstacle message...')
                        obc_pub.publish(obc_msg)
                        obs_pub.publish(obs_msg)
                        obs_i_pub.publish(obs_i_msg)
                        length_pub.publish(length_msg)
                        rospy.sleep(0.5)
                pub_thread = threading.Thread(target=publisher, args=())
                pub_thread.start()

                print('waiting for lightning service...')
                try:
                    # to make sure when we CTRL+C we can exit
                    rospy.wait_for_service(LIGHTNING_SERVICE)
                except:
                    exception = True
                    pub_thread.join()
                    # exit because there is error
                    print('exception occurred when waiting for lightning service...')
                    raise
                print('acquired lightning service')
                lightning = rospy.ServiceProxy(LIGHTNING_SERVICE, GetMotionPlan)
                try:
                    respond = lightning(request)
                except:
                    exception = True
                responded = True
                pub_thread.join()

                if respond.motion_plan_response.error_code.val == respond.motion_plan_response.error_code.SUCCESS:
                    # succeed
                    time = respond.motion_plan_response.planning_time
                    time_path.append(time)
                    path = respond.motion_plan_response.trajectory.joint_trajectory.points
                    path = [p.positions for p in path]
                    path = np.array(path)
                    print(path)
                    # feasibility check this path
                    if plan_general.feasibility_check(path, obc, IsInCollision, step_sz=0.01):
                        fp = 1
                        print('feasible')
                    else:
                        fp = 0
                        print('not feasible')
            fes_path.append(fp)
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print(np.sum(np.array(fes_env)))
        print('accuracy up to now: %f' % (float(np.sum(np.array(fes_env))) / np.sum(np.array(valid_env))))
    pickle.dump(time_env, open(args.res_path+'time.p', "wb" ))
    f = open(os.path.join(args.res_path,'accuracy.txt'), 'w')
    valid_env = np.array(valid_env).flatten()
    fes_env = np.array(fes_env).flatten()   # notice different environments are involved
    suc_rate = float(fes_env.sum()) / valid_env.sum()
    f.write(str(suc_rate))
    f.close()

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
