from experiments.simple data_loader_2d, data_loader_r2d, data_loader_r3d
import argparse
import pickle
import sys
import time
import os
import numpy as np
import rospy
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/root/catkin_ws/src/lightning_ros/scripts')
from lightning.msg import Float64Array, Float64Array2D
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest, GetMotionPlanResponse
from moveit_msgs.msg import JointConstraint, Constraints
from std_msgs.msg import String, Float64
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
        time_limit = 60.
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
        time_limit = 60.
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
    length_pub = rospy.Publisher('lightning/planning/path_length_threshold', Float64, queue_size=10)

    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        obc = obcs[i]
        # publishing to ROS topic
        obc_msg = Float64Array2D([Float64Array(obci) for obci in obc])
        obs_msg = Float64Array(obs[i])
        for j in range(len(paths[0])):
            if path_lengths[i][j] == 0:
                continue
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            else:
                valid_path.append(1)
                # obtaining the length threshold for planning
                data_length = 0.
                for k in range(path_lengths[i][j]-1):
                    data_length += np.linalg.norm(paths[i][j][k+1]-paths[i][j][k])
                length_msg = Float64(data_length * ratio)
                # call lightning service
                print('waiting for lightning service...')
                rospy.wait_for_service(LIGHTNING_SERVICE)
                print('acquired lightning service')
                lightning = rospy.ServiceProxy(LIGHTNING_SERVICE, GetMotionPlan)
                request = GetMotionPlanRequest()
                request.motion_plan_request.group_name = 'base'
                for k in range(len(paths[i][j][0])):
                    request.motion_plan_request.start_state.joint_state.name.append('%d' % (k))
                    request.motion_plan_request.start_state.joint_state.position.append(paths[i][j][0][k])

                request.motion_plan_request.goal_constraints.append(Constraints())
                for k in range(len(paths[i][j][0])):
                    request.motion_plan_request.goal_constraints[0].joint_constraints.append(JointConstraint())
                    request.motion_plan_request.goal_constraints[0].joint_constraints[k].position = paths[i][j][path_lengths[i][j]-1][k]
                request.motion_plan_request.allowed_planning_time = time_limit

                responded = False
                exception = False
                def publisher():
                    global responded, exception
                    while not responded and not exception:
                        obc_pub.publish(obc_msg)
                        obs_pub.publish(obs_msg)
                        length_pub.publish(length_msg)
                pub_thread = threading.Thread(target=publisher, args=())

                pub_thread.start()
                try:
                    respond = lightning(request)
                except:
                    exception = True
                responded = True
                pub_thread.join()

                if respond.motion_plan_response.error_code.val == respond.motion_plan_response.error_code.SUCCESS:
                    # succeed
                    fp = 1
                    time = respond.motion_plan_response.planning_time
                    time_path.append(time)
                    print('feasible')
            fes_path.append(fp)
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (np.sum(np.array(fes_env)) / np.sum(np.array(valid_env))))
    pickle.dump(time_env, open(args.model_path+'time.p', "wb" ))
    f = open(os.path.join(args.model_path,'accuracy.txt'), 'w')
    valid_env = np.array(valid_env).flatten()
    fes_env = np.array(fes_env).flatten()   # notice different environments are involved
    seen_test_suc_rate = fes_env.sum() / valid_env.sum()
    f.write(str(seen_test_suc_rate))
    f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../lightning_res/')
parser.add_argument('--env_idx', type=int, default=0, help='from which env')
parser.add_argument('--path_idx', type=int, default=0, help='from which path')
parser.add_argument('--N', type=int, default=1)
parser.add_argument('--NP', type=int, default=10)

parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--env_type', type=str, default='s2d')
args = parser.parse_args()
plan(args)
