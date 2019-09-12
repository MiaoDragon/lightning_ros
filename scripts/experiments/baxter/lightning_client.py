import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
#sys.path.insert(1, '/home/yinglong/Documents/MotionPlanning/baxter/ros_ws/src/lightning_ros/scripts')
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
from planning_scene_editor import *
from get_state_validity import StateValidity
from import_tool import fileImport
from path_data_loader import load_test_dataset_end2end
import utility_baxter
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
joint_ranges = [3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67]
def IsInCollision(state, print_depth=False):
    # returns true if robot state is in collision, false if robot state is collision free
    filler_robot_state[10:17] = moveit_scrambler(np.multiply(state,joint_ranges))
    rs_man.joint_state.position = tuple(filler_robot_state)
    collision_free = sv.getStateValidity(rs_man, group_name="right_arm", print_depth=print_depth)
    return (not collision_free)



def plan(args):
    global responded, exception, sv, rs_man, filler_robot_state
    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path

    envs = importer.environments_import(env_data_path + args.envs_file)
    with open (env_data_path+args.envs_file, 'rb') as env_f:
        envDict = pickle.load(env_f)

    obstacles, paths, path_lengths = load_test_dataset_end2end(envs, path_data_path, pcd_data_path, args.path_data_file, importer, NP=args.NP)
    obstacles = obstacles[args.env_idx:args.env_idx+args.N]
    # remember to change the loading NP above to be all path
    paths = paths[args.env_idx:args.env_idx+args.N, args.path_idx:args.path_idx+args.NP]
    path_lengths = path_lengths[args.env_idx:args.env_idx+args.N, args.path_idx:args.path_idx+args.NP]
    rospy.init_node('lightning_client')
    print('loading...')

    #rospy.init_node("environment_monitor")
    scene = PlanningSceneInterface()
    robot = RobotCommander()
    group = MoveGroupCommander("right_arm")
    scene._scene_pub = rospy.Publisher('planning_scene',
                                        PlanningScene,
                                        queue_size=0)

    sv = StateValidity()
    set_environment(robot, scene)

    masterModifier = ShelfSceneModifier()
    sceneModifier = PlanningSceneModifier(envDict['obsData'])
    sceneModifier.setup_scene(scene, robot, group)
    rs_man = RobotState()
    robot_state = robot.get_current_state()
    rs_man.joint_state.name = robot_state.joint_state.name
    filler_robot_state = list(robot_state.joint_state.position)

    dof=7

    tp=0
    fp=0

    et_tot = []
    neural_paths = {}
    bad_paths = {}

    goal_collision = []

    if not os.path.exists(args.good_path_sample_path):
        os.makedirs(args.good_path_sample_path)
    if not os.path.exists(args.bad_path_sample_path):
        os.makedirs(args.bad_path_sample_path)


    experiment_name = args.experiment_name
    good_paths_path = args.good_path_sample_path + '/' + experiment_name
    bad_paths_path = args.bad_path_sample_path + '/' + experiment_name
    obs_pub = rospy.Publisher('lightning/obstacles/obs', Float64Array, queue_size=10)
    obs_i_pub = rospy.Publisher('lightning/obstacles/obs_i', Int32, queue_size=10)
    length_pub = rospy.Publisher('lightning/planning/path_length_threshold', Float64, queue_size=10)
    obs = obstacles

    for i, env_name in enumerate(envs):
        et=[]
        col_env = []
        tp_env = 0
        fp_env = 0
        neural_paths[env_name] = []
        bad_paths[env_name] = []

        if not os.path.exists(good_paths_path + '/' + env_name):
            os.makedirs(good_paths_path + '/' + env_name)
        if not os.path.exists(bad_paths_path + '/' + env_name):
            os.makedirs(bad_paths_path + '/' + env_name)

        print("ENVIRONMENT: " + env_name)

        sceneModifier.delete_obstacles()
        new_pose = envDict['poses'][env_name]
        sceneModifier.permute_obstacles(new_pose)
        # ROS message of obstacle information
        print(obstacles[i])
        obs_msg = Float64Array(obstacles[i])
        obs_i_msg = Int32(i)
        for j in range(0,path_lengths.shape[1]):
            print ("step: i="+str(i)+" j="+str(j))
            print("fp: " + str(fp_env))
            print("tp: " + str(tp_env))

            obs=obstacles[i]

            if path_lengths[i][j]>0:
                print(path_lengths[i][j])
                start=np.zeros(dof,dtype=np.float32)
                goal=np.zeros(dof,dtype=np.float32)
                for l in range(0,dof):
                    start[l]=paths[i][j][0][l]

                for l in range(0,dof):
                    goal[l]=paths[i][j][path_lengths[i][j]-1][l]
                if (IsInCollision(goal)):
                    print("GOAL IN COLLISION --- BREAKING")
                    goal_collision.append(j)
                    continue
                print("unnormalizing and scrambling to MoveIt format...")
                start = utility_baxter.unnormalize(start, joint_ranges)
                goal = utility_baxter.unnormalize(goal, joint_ranges)
                # calculate path length
                path = np.array(paths[i][j])
                #path = utility_baxter.unnormalize(path, joint_ranges)
                for k in range(len(path)):
                    path[k] = utility_baxter.unnormalize(path[k],joint_ranges)
                data_length = 0
                for k in xrange(path_lengths[i][j]-1):
                    data_length += np.linalg.norm(path[k+1]-path[k])
                tp=tp+1
                tp_env=tp_env+1
                # call lightning framework
                ratio = 1.
                time_limit = 30
                length_msg = Float64(data_length * ratio)
                # call lightning service
                request = GetMotionPlanRequest()
                request.motion_plan_request.group_name = 'right_arm'
                request.motion_plan_request.start_state.joint_state.name = robot_state.joint_state.name[10:17]
                filler_robot_state[10:17] = start
                #request.motion_plan_request.start_state.joint_state.position = tuple(filler_robot_state)
                request.motion_plan_request.start_state.joint_state.position = tuple(start)
                filler_robot_state[10:17] = goal
                request.motion_plan_request.goal_constraints.append(Constraints())
                #for k in xrange(len(filler_robot_state)):
                for k in xrange(len(goal)):
                    request.motion_plan_request.goal_constraints[0].joint_constraints.append(JointConstraint())
                    #request.motion_plan_request.goal_constraints[0].joint_constraints[k].position = filler_robot_state[k]
                    request.motion_plan_request.goal_constraints[0].joint_constraints[k].position = goal[k]
                request.motion_plan_request.allowed_planning_time = time_limit

                responded = False
                exception = False
                def publisher():
                    global responded, exception
                    while not responded and not exception:
                        print('sending obstacle message...')
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
                    fp_env += 1
                    fp += 1
                    time = respond.motion_plan_response.planning_time
                    et.append(time)
                    print('feasible')
                    path = respond.motion_plan_response.trajectory.joint_trajectory.points
                    path = [p.positions for p in path]
                    with open(good_paths_path + '/' + env_name + '/fp_%d.' % (j+args.path_idx) + 'pkl', 'wb') as path_f:
                        pickle.dump(path, path_f)
                    neural_paths[env_name].append(path)

        et_tot.append(et)
        print("total env paths: ")
        print(tp_env)
        print("feasible env paths: ")
        print(fp_env)
        print("average time: ")
        print(np.mean(et))
        env_data = {}
        env_data['tp_env'] = tp_env
        env_data['fp_env'] = fp_env
        env_data['et_env'] = et
        # we already save the path before, so no need to save it again
        #env_data['paths'] = neural_paths[env_name]

        with open(good_paths_path + '/' + env_name + '/env_data.pkl', 'wb') as data_f:
            pickle.dump(env_data, data_f)

    print("total paths: ")
    print(tp)
    print("feasible paths: ")
    print(fp)

    with open(good_paths_path+'neural_paths.pkl', 'wb') as good_f:
        pickle.dump(neural_paths, good_f)

    with open(good_paths_path+'elapsed_time.pkl', 'wb') as time_f:
        pickle.dump(et_tot, time_f)

    print(np.mean([np.mean(x) for x in et_tot]))
    print(np.std([np.mean(x) for x in et_tot]))

    acc = []

    for i, env in enumerate(envs):
        with open (good_paths_path+env+'_env_data.pkl', 'rb') as data_f:
            data = pickle.load(data_f)
        acc.append(100.0*data['fp_env']/data['tp_env'])
        print("env: " + env)
        print("accuracy: " + str(100.0*data['fp_env']/data['tp_env']))
        print("time: " + str(np.mean(data['et_env'])))
        print("min time: " + str(np.min(data['et_env'])))
        print("max time: " + str(np.max(data['et_env'])))
        print("\n")

    print(np.mean(acc))
    print(np.std(acc))

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=1)
parser.add_argument('--NP', type=int, default=10)
parser.add_argument('--env_idx', type=int, default=0)
parser.add_argument('--path_idx', type=int, default=0)


parser.add_argument('--env_data_path', type=str, default='')
parser.add_argument('--path_data_path', type=str, default='')
parser.add_argument('--path_data_file', type=str, default='')
parser.add_argument('--pointcloud_data_path', type=str, default='')
parser.add_argument('--envs_file', type=str, default='')
parser.add_argument('--good_path_sample_path', type=str, default='./path_samples/good_path_samples')
parser.add_argument('--bad_path_sample_path', type=str, default='./path_samples/bad_path_samples')
parser.add_argument('--experiment_name', type=str, default='test_experiment')
# whether to save path or not
parser.add_argument('--save_path', type=int, default=0)
args = parser.parse_args()
plan(args)
