#!/usr/bin/env python
"""
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, University of California, Berkeley
# All rights reserved.
# Authors: Cameron Lee (cameronlee@berkeley.edu) and Dmitry Berenson (
berenson@eecs.berkeley.edu)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of University of California, Berkeley nor the names
of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""

"""
This node is the top-level node for the lightning library. It advertises a
service, "lightning_get_path", of type GetMotionPlan, and uses the RR and PFS
nodes to retrieve plans. Whichever node (RR or PFS) completes first has its
path returned while the other node is stopped.
This node also performs some post-processing on the RR retrieved paths to
smooth out the path. This post-processing is not strictly necessary; all it
does is shortcut between random pairs of points.
"""
import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import roslib
#roslib.load_manifest("lightning");
import rospy
import actionlib
import threading

import time
import os
from lightning.msg import UpdateAction, UpdateGoal
from lightning.msg import RRAction, RRGoal, PFSAction, PFSGoal, Float64Array, StopPlanning, Stats, PlannerType
from lightning.srv import ManagePathLibrary, ManagePathLibraryRequest, PathShortcut, PathShortcutRequest
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from std_msgs.msg import Float32, UInt8
from trajectory_msgs.msg import JointTrajectoryPoint
from tools import NeuralOMPLPathTools, NeuralPathTools
from tools import utility
from experiments.simple import plan_general
from architecture.GEM_end2end_model import End2EndMPNet

import numpy as np
import torch
# Node names for RR and PFS plan retrieval.
RR_NODE_NAME = "rr_node"
PFS_NODE_NAME = "pfs_node"
# Topic names for stopping RR and PFS nodes.
STOP_RR_NAME = "stop_all_rr"
STOP_PFS_NAME = "stop_all_pfs"
# Name to advertise service on. Service advertised from this file.
LIGHTNING_SERVICE = "lightning_get_path"
# Service for retrieving currnt PlanningScene from MoveIt.
SET_PLANNING_SCENE_DIFF_NAME = "/get_planning_scene";
# Service name for managing path library.
MANAGE_LIBRARY = "manage_path_library"

# Topic to publish to for updating the neural network
# UPDATE_TOPIC = 'model_update'
class Lightning:
    def __init__(self):
        # depending on the argument framework_type, use different classes and settings
        framework_type = rospy.get_param('framework_type')
        if framework_type == 'ompl':
            ShortcutPathWrapper = NeuralOMPLPathTools.ShortcutPathWrapper
            DrawPointsWrapper = NeuralOMPLPathTools.DrawPointsWrapper
        elif framework_type == 'moveit':
            # Make sure that the moveit server is ready before starting up
            rospy.wait_for_service(PLANNING_SCENE_SERV_NAME);
            rospy.wait_for_service(SET_PLANNING_SCENE_DIFF_NAME); #make sure the environment server is ready before starting up
            ShortcutPathWrapper = NeuralOMPLPathTools.ShortcutPathWrapper
            DrawPointsWrapper = NeuralOMPLPathTools.DrawPointsWrapper

        # Initialize clients for planners.
        self.rr_client = actionlib.SimpleActionClient(RR_NODE_NAME, RRAction)
        self.pfs_client = actionlib.SimpleActionClient(PFS_NODE_NAME, PFSAction)
        # Client for managing path library.
        self.manage_library_client = rospy.ServiceProxy(MANAGE_LIBRARY, ManagePathLibrary)
        self.store_paths = rospy.get_param("~store_paths")
        self.use_rr = rospy.get_param("~use_RR")
        self.use_pfs = rospy.get_param("~use_PFS")
        if not self.use_rr and not self.use_pfs:
            rospy.logerr("Lightning: at least one of use_RR and use_PFS need to be true")
        self.shortcut_path_wrapper = ShortcutPathWrapper()
        self.lightning_response = None
        self.lightning_service = rospy.Service(LIGHTNING_SERVICE, GetMotionPlan, self.run)
        self.current_joint_names = []
        self.current_group_name = ""
        self.robot_name = rospy.get_param("robot_name")
        self.stop_rr_publisher = rospy.Publisher(STOP_RR_NAME, StopPlanning, queue_size=10)
        self.stop_pfs_publisher = rospy.Publisher(STOP_PFS_NAME, StopPlanning, queue_size=10)
        self.rr_returned, self.pfs_returned = False, False
        self.done_lock = threading.Lock()
        self.lightning_response_ready_event = threading.Event()
        #if draw_points is True, then display points in rviz
        self.draw_points = rospy.get_param("draw_points")
        if self.draw_points:
            self.draw_points_wrapper = DrawPointsWrapper()

        self.publish_stats = rospy.get_param("publish_stats")
        if self.publish_stats:
          # Publishes statistics having to do with the time it takes various
          # parts of the lightning planner to run.
          self.stat_pub = rospy.Publisher("stats", Stats, queue_size=10)


        self.normalize_func = utility.get_normalizer()
        #### neural network setup
        # construct model from fed environment
        self.data_all = []
        self.num_path_trained = 0
        self.num_trained_samples = 0
        self.model_path = rospy.get_param('model/model_path')
        self.model_name = rospy.get_param('model/model_name')
        self.freq_rehersal = rospy.get_param('model/freq_rehersal')
        self.batch_rehersal = rospy.get_param('model/batch_rehersal')
        self.freq_save = rospy.get_param('model/freq_save')
        self.plan_times = []
        self.plan_time = 0.
        # set device (CUDA or CPU)
        device_name = rospy.get_param('model/server_device')
        device = torch.device(device_name)
        self.device = device
        if device_name != 'cpu':
            torch.cuda.set_device(int(device_name.split(':')[1]))
        self.model = utility.create_and_load_model(End2EndMPNet, self.model_path+self.model_name, device)
        self.retrieved_and_final_path = [None, None, None, None, None, None]
        # record current planning path
        # format: [planner_type, path, planner_type, path]

        self.torch_seed, self.np_seed, self.py_seed = 0, 0, 0

        # to make sure initially the planner and server have the same model,
        # we save the model at the beginning if the model file does not exist
        if not os.path.isfile(self.model_path+self.model_name):
            rospy.loginfo('Lightning: Saving initial network parameters...')
            utility.save_state(self.model, self.torch_seed, self.np_seed, self.py_seed, self.model_path+self.model_name)
        # notify to synchronize model weights
        # create client to signal planner for updating
        self.pfs_update_client = actionlib.SimpleActionClient('/lightning/'+PFS_NODE_NAME+'/update', UpdateAction)
        self.rr_update_client = actionlib.SimpleActionClient('/lightning/'+RR_NODE_NAME+'/update', UpdateAction)
        ## TODO: use conditional lock for faster response
        # this lock is for preventing save and load at the same time
        #self.model_lock = threading.Lock()
        #self.read_or_write = -1  # -1 for idle, 0 for write, 1+ for read
        rospy.loginfo('Lightning: Notify planner to update network...')
        self._notify_update('pfs')
        self._notify_update('rr')

        # for information of training
        self.losses = []
        self.total_num_paths = []  # this record how many paths are planned
        self.total_num_paths_NN = []   # this record how many are planned by NN
        self.time = []
        self.plan_mode = []  # either PFS or RR
        self.total_new_nodes = []  # total number of newly generated nodes
        self.total_new_nodes_NN = []  # total number of newly generated nodes by NN
        self.total_new_node = 0
        self.total_new_node_NN = 0


    def _notify_update(self, client_name):
        if client_name == 'pfs':
            client = self.pfs_update_client
        else:
            client = self.rr_update_client
        update_client_goal = UpdateGoal()
        client.wait_for_server()
        """
        ### TODO:  below is a conditional lock so that the lightning server can be faster
        # not sure if bug-free
        while True:
            # check the model lock to prevent writing when reading
            self.model_lock.acquire()
            if self.read_or_write == -1:
                self.read_or_write = 1
            if self.read_or_write >= 1:
                # one more reader
                self.read_or_write += 1
            if self.read_or_write >= 1:
                self.model_lock.release()
                break
            self.model_lock.release()
        """
        rospy.loginfo("Lightning: Sending Update Goal to %s Client" % (client_name))
        # a safer way: when model is updating, we just wait for it to complete
        client.send_goal(update_client_goal)
        client.wait_for_result()

    # Called in a separate thread to stop the planning nodes in case of timeout.
    def _lightning_timeout(self, time):
        self.lightning_response_ready_event.wait(time)
        if self.lightning_response is None:
            rospy.loginfo("Lightning: ran out of time")
            if self.use_rr:
                self._send_stop_rr_planning()
            if self.use_pfs:
                self._send_stop_pfs_planning()

    # Main service routine advertised for the benefit of the user.
    def run(self, request):
        #make sure the request is valid
        self.retrieved_and_final_path = [None, None, None, None, None, None]
        self.total_new_node = 0
        self.total_new_node_NN = 0
        start_and_goal = self._is_valid_motion_plan_request(request)
        if start_and_goal is None:
            response = GetMotionPlanResponse()
            response.motion_plan_response.error_code.val = response.motion_plan_response.error_code.PLANNING_FAILED
            return response
        s, g = start_and_goal

        self.rr_returned, self.pfs_returned = False, False
        self.lightning_response = None
        self.lightning_response_ready_event.clear()
        self.current_joint_names = request.motion_plan_request.start_state.joint_state.name
        self.current_group_name = request.motion_plan_request.group_name

        if self.draw_points:
            self.draw_points_wrapper.clear_points()

        #start a timer that stops planners if they take too long
        timer = threading.Thread(target=self._lightning_timeout, args=(request.motion_plan_request.allowed_planning_time,))
        timer.start()

        self.start_time = time.time() # Used for timing stats.
        # Send action requests to RR and PFS nodes.
        if self.use_rr:
            rr_client_goal = RRGoal()
            rr_client_goal.start = s
            rr_client_goal.goal = g
            rr_client_goal.joint_names = self.current_joint_names
            rr_client_goal.group_name = self.current_group_name
            rr_client_goal.allowed_planning_time = rospy.Duration(
                request.motion_plan_request.allowed_planning_time)
            self.rr_client.wait_for_server()
            rospy.loginfo("Lightning: Sending goal to RR")
            self.rr_client.send_goal(rr_client_goal, done_cb=self._rr_done_cb)

        if self.use_pfs:
            pfs_client_goal = PFSGoal()
            pfs_client_goal.start = s
            pfs_client_goal.goal = g
            pfs_client_goal.joint_names = self.current_joint_names
            pfs_client_goal.group_name = self.current_group_name
            pfs_client_goal.allowed_planning_time = rospy.Duration(
                request.motion_plan_request.allowed_planning_time)
            self.pfs_client.wait_for_server()
            rospy.loginfo("Lightning: Sending goal to PFS")
            self.pfs_client.send_goal(pfs_client_goal, done_cb=self._pfs_done_cb)

        self.lightning_response_ready_event.wait()
        if self.lightning_response.motion_plan_response.error_code.val != self.lightning_response.motion_plan_response.error_code.SUCCESS:
            rospy.loginfo("Lightning: did not find a path")
            # no need to do train MPNet
            return self.lightning_response

        rospy.loginfo("Lightning: Lightning is responding with a path")
        """
        depending on the path property, train MPNet or not.
        If the path is planned by Classical planner before or now, train it with MPNet.
        1. transform path data into pytorch version
        2. train Continual MPNet with the new path
        3. save the new model weights to file
        4. notify planners to update the model
        """
        # transform path into pytorch version
        self.train_model()
        return self.lightning_response

    def train_model(self):
        """
        depending on the path property, train MPNet or not.
        If the path is planned by Classical planner before or now, train it with MPNet.
        1. transform path data into pytorch version
        2. train Continual MPNet with the new path
        3. save the new model weights to file
        4. notify planners to update the model
        """
        retrieved_planner_type, retrieved_path, final_planner_type, final_path, num_paths, num_NN_paths = self.retrieved_and_final_path
        # record stats
        self.total_num_paths.append(num_paths)
        self.total_num_paths_NN.append(num_NN_paths)
        self.total_new_nodes.append(self.total_new_node)
        self.total_new_nodes_NN.append(self.total_new_node_NN)
        self.plan_times.append(self.plan_time)
        if retrieved_planner_type is None:
            self.plan_mode.append(0)  # 0 for pfs, 1 for rr
        else:
            self.plan_mode.append(1)
        # depending on retrieved_planner_type and final_planner, train the network
        if (retrieved_planner_type is None and final_planner_type == PlannerType.NEURAL) \
            or (retrieved_planner_type == PlannerType.NEURAL and final_planner_type == PlannerType.NEURAL):
            to_save = {}
            to_save['loss'] = self.losses
            to_save['total_num_paths'] = self.total_num_paths
            to_save['total_num_paths_NN'] = self.total_num_paths_NN
            to_save['plan_time'] = self.plan_time
            to_save['plan_mode'] = self.plan_mode
            to_save['total_new_node'] = self.total_new_nodes
            to_save['total_new_node_NN'] = self.total_new_nodes_NN
            utility.save_info(to_save, self.model_path+'lightning_res.pkl')
            return
        rospy.loginfo('Lightning: Training Neural Network...')
        # receive obstacle information
        obs = rospy.wait_for_message('obstacles/obs', Float64Array)
        obs = obs.values
        obs = torch.FloatTensor(obs)

        dataset, targets, env_indices = plan_general.transformToTrain(final_path, len(final_path), obs, 0)
        self.data_all += list(zip(dataset, targets, env_indices))
        self.num_trained_samples += len(targets)
        added_data = list(zip(dataset,targets,env_indices))
        bi = np.concatenate( (obs.numpy().reshape(1,-1).repeat(len(dataset),axis=0), dataset), axis=1).astype(np.float32)
        bi = self.normalize_func(bi)
        targets = self.normalize_func(targets)
        bi = torch.FloatTensor(bi)
        bt = torch.FloatTensor(targets)
        self.model.zero_grad()
        bi=utility.to_var(bi, self.device)
        bt=utility.to_var(bt, self.device)
        self.model.observe(bi, 0, bt)
        self.num_path_trained += 1
        # rehersal
        if self.num_path_trained % self.freq_rehersal == 0 and len(self.data_all) > self.batch_rehersal:
            rospy.loginfo('Lightning: Rehersal...')
            print('rehersal...')
            sample = random.sample(self.data_all, self.batch_rehersal)
            dataset, targets, env_indices = list(zip(*sample))
            dataset, targets, env_indices = list(dataset), list(targets), list(env_indices)
            bi = np.concatenate( (obs[env_indices], dataset), axis=1).astype(np.float32)
            bt = targets
            bi = torch.FloatTensor(bi)
            bt = torch.FloatTensor(bt)
            bi, bt = self.normalize_func(bi), self.normalize_func(bt)
            self.model.zero_grad()
            bi=utility.to_var(bi, self.device)
            bt=utility.to_var(bt, self.device)
            self.model.observe(bi, 0, bt, False) # train but don't remember
        # obtain the loss after training:
        loss = self.model.loss(self.model.forward(bi), bt)
        loss = loss.data.cpu()
        print("loss: %f" % (loss))
        print('planner type: %d' % (final_planner_type))
        self.losses.append(loss.data.cpu().item())

        if self.num_path_trained % self.freq_save == 0:
            # save loss and planner type
            to_save = {}
            to_save['loss'] = self.losses
            to_save['total_num_paths'] = self.total_num_paths
            to_save['total_num_paths_NN'] = self.total_num_paths_NN
            to_save['plan_time'] = self.plan_time
            to_save['plan_mode'] = self.plan_mode
            to_save['total_new_node'] = self.total_new_nodes
            to_save['total_new_node_NN'] = self.total_new_nodes_NN
            utility.save_info(to_save, self.model_path+'lightning_res.pkl')

        # write trained model to file
        utility.save_state(self.model, self.torch_seed, self.np_seed, self.py_seed, self.model_path+self.model_name)
        # notify planners to update the model
        msg = UInt8(0)
        rospy.loginfo('Lightning: Notify planner to update network...')
        self._notify_update('pfs')
        self._notify_update('rr')

    def _print_error(self, msg):
        rospy.logerr("***ERROR*** %s ***ERROR***" % (msg))

    # Check to ensure that the request we received contains all of the
    # components that we might need, namely that timeout, the constraints,
    # and a start/goal state.
    def _is_valid_motion_plan_request(self, request):
        if request.motion_plan_request.allowed_planning_time <= 0:
            self._print_error("Lightning: requires allowed_planning_time to be greater than 0")
            return None

        if len(request.motion_plan_request.goal_constraints[0].position_constraints) > 0:
            self._print_error("Lightning: does not handle position constraints")
            return None

        s = list(request.motion_plan_request.start_state.joint_state.position)
        g = []
        for jc in request.motion_plan_request.goal_constraints[0].joint_constraints:
            if jc.tolerance_above != 0 or jc.tolerance_below != 0:
                self._print_error("Lightning: does not handle tolerances")
                return None
            else:
                g.append(jc.position)

        if len(s) == 0:
            self._print_error("Lightning: did not receive a start state")
            return None

        if len(g) == 0:
            self._print_error("Lightning: did not receive a goal state")
            return None
        return s, g

    # Called if RR finishes before PFS; stops the PFS and returns the retrieved
    # path, so long as the planner succeeds.
    def _rr_done_cb(self, state, result):
        self.done_lock.acquire()
        self.rr_returned = True
        if self.publish_stats:
          stat_msg = Stats()
          stat_msg.plan_time = time.time() - self.start_time
        if result.status.status == result.status.SUCCESS:
            if not self.pfs_returned or self.lightning_response is None:
                self._send_stop_pfs_planning()

                rr_path = [p.values for p in result.repaired_path]
                retrieved_path = [p.values for p in result.retrieved_path]
                total_num_paths = result.total_num_paths
                total_num_paths_NN = result.total_num_paths_NN
                self.total_new_node = result.total_new_node
                self.total_new_node_NN = result.total_new_node_NN
                print('rr_done_cb: total_num_paths_NN: %d' % (total_num_paths_NN))
                shortcut_start = time.time()
                shortcut = self.shortcut_path_wrapper.shortcut_path(rr_path, self.current_group_name)
                if self.publish_stats:
                  stat_msg.shortcut_time = time.time() - shortcut_start

                self.lightning_response = self._create_get_motion_plan_response(shortcut)
                # set planning time to be the total time up to now
                self.lightning_response.motion_plan_response.planning_time = time.time() - self.start_time
                # record the planned path and planner
                self.retrieved_and_final_path = [result.retrieved_planner_type.planner_type, None, \
                                                 result.repaired_planner_type.planner_type, rr_path, \
                                                 total_num_paths, total_num_paths_NN]

                self.lightning_response_ready_event.set()
                self.done_lock.release()

                #display new path in rviz
                if self.draw_points:
                    self.draw_points_wrapper.draw_points(rr_path, self.current_group_name, "final", DrawPointsWrapper.ANGLES, DrawPointsWrapper.GREEN, 0.1)

                if self.store_paths:
                    #store_response = self._store_path(rr_path, rr_path) # this original version seems to be wrong because retrieved is the same as repaired
                    store_response = self._store_path(rr_path, retrieved_path, result.repaired_planner_type.planner_type)
                    self._special_print("Lightning: Got a path from RR, path stored = %s, number of library paths = %i" % (store_response))
                else:
                    self._special_print("Lightning: Got a path from RR")
                # record the planning time
                self.plan_time = time.time() - self.start_time
                if self.publish_stats:
                  stat_msg.time = time.time() - self.start_time
                  stat_msg.rr_won = True
                  self.stat_pub.publish(stat_msg)
                return
        else:
            rospy.loginfo("Lightning: Call to RR did not return a path")
            if not self.use_pfs or (self.pfs_returned and self.lightning_response is None):
                self.lightning_response = GetMotionPlanResponse()
                self.lightning_response.motion_plan_response.error_code.val = self.lightning_response.motion_plan_response.error_code.PLANNING_FAILED
                self.lightning_response_ready_event.set()
        self.done_lock.release()

    # Called if PFS finishes before RR; stops the RR and returns the retrieved
    # path, so long as the planner succeeds.
    def _pfs_done_cb(self, state, result):
        self.done_lock.acquire()
        self.pfs_returned = True
        if self.publish_stats:
          stat_msg = Stats()
          stat_msg.rr_won = False
          stat_msg.plan_time = time.time() - self.start_time
        if result.status.status == result.status.SUCCESS:
            if not self.rr_returned or self.lightning_response is None:
                self._send_stop_rr_planning()

                pfsPath = [p.values for p in result.path]
                shortcut_start = time.time()
                shortcut = self.shortcut_path_wrapper.shortcut_path(pfsPath, self.current_group_name)
                if self.publish_stats:
                  stat_msg.shortcut_time = time.time() - shortcut_start

                self.lightning_response = self._create_get_motion_plan_response(shortcut)
                # set planning time to be the total time up to now
                self.lightning_response.motion_plan_response.planning_time = time.time() - self.start_time

                # record the planned path and planner
                self.retrieved_and_final_path = [None, None, result.planner_type.planner_type, pfsPath]
                if result.planner_type.planner_type == PlannerType.CLASSIC:
                    self.retrieved_and_final_path += [1, 0]
                    # store the number of newly generated nodes
                    self.total_new_node = len(pfsPath)
                else:
                    self.retrieved_and_final_path += [1, 1]
                    # store the number of newly generated nodes
                    self.total_new_node = len(pfsPath)
                    self.total_new_node_NN = len(pfsPath)

                self.lightning_response_ready_event.set()
                self.done_lock.release()

                #display new path in rviz
                if self.draw_points:
                    self.draw_points_wrapper.draw_points(pfsPath, self.current_group_name, "final", DrawPointsWrapper.ANGLES, DrawPointsWrapper.GREEN, 0.1)

                if self.store_paths:
                    store_response = self._store_path(pfsPath, [], result.planner_type.planner_type)
                    self._special_print("Lightning: Got a path from PFS, path stored = %s, number of library paths = %i" % (store_response))
                else:
                    self._special_print("Lightning: Got a path from PFS")
                self.plan_time = time.time() - self.start_time
                if self.publish_stats:
                  stat_msg.time = time.time() - self.start_time
                  self.stat_pub.publish(stat_msg)
                return
        else:
            rospy.loginfo("Lightning: Call to PFS did not return a path")
            if not self.use_rr or (self.rr_returned and self.lightning_response is None):
                self.lightning_response = GetMotionPlanResponse()
                self.lightning_response.motion_plan_response.error_code.val = self.lightning_response.motion_plan_response.error_code.PLANNING_FAILED
                self.lightning_response_ready_event.set()
        self.done_lock.release()

    # Takes path returned from PFS/RR and puts it in appropriate form to
    # send back to the user.
    def _create_get_motion_plan_response(self, path):
        response = GetMotionPlanResponse()
        response.motion_plan_response.error_code.val = response.motion_plan_response.error_code.SUCCESS
        response.motion_plan_response.trajectory.joint_trajectory.points = []
        for pt in path:
            jtp = JointTrajectoryPoint()
            jtp.positions = pt
            response.motion_plan_response.trajectory.joint_trajectory.points.append(jtp)
        response.motion_plan_response.trajectory.joint_trajectory.joint_names = self.current_joint_names
        return response

    # Calls PathTools library to store most recent path.
    def _store_path(self, final_path, retrieved_path, planner_type):
        store_request = ManagePathLibraryRequest()
        store_request.joint_names = self.current_joint_names
        store_request.robot_name = self.robot_name
        store_request.action = store_request.ACTION_STORE
        store_request.planner_type = planner_type
        # convert into apporpriate request.
        for point in final_path:
            jtp = JointTrajectoryPoint()
            jtp.positions = point
            store_request.path_to_store.append(jtp)
        for point in retrieved_path:
            jtp = JointTrajectoryPoint()
            jtp.positions = point
            store_request.retrieved_path.append(jtp)
        # wait for storing service
        rospy.wait_for_service(MANAGE_LIBRARY);
        store_response = self.manage_library_client(store_request)
        return (store_response.path_stored, store_response.num_library_paths)

    def _send_stop_pfs_planning(self):
        self.stop_pfs_publisher.publish(self._create_stop_planning_message())

    def _send_stop_rr_planning(self):
        self.stop_rr_publisher.publish(self._create_stop_planning_message())

    def _create_stop_planning_message(self):
        stop_message = StopPlanning()
        stop_message.planner_id = rospy.get_param("planner_config_name")
        stop_message.group_name = self.current_group_name
        return stop_message

    def _special_print(self, msg):
        rospy.loginfo("**************")
        rospy.loginfo(msg)
        rospy.loginfo("**************")

if __name__ == "__main__":
    try:
        rospy.init_node("lightning");
        light = Lightning();
        rospy.loginfo("Lightning: ready")
        rospy.spin();
    except rospy.ROSInterruptException:
        pass;
