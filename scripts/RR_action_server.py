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
This node advertises an action which is used by the main lightning node
(see run_lightning.py) to run the Retrieve and Repair portion of LightningROS.
This node relies on a planner_stoppable type node to repair the paths, the
PathTools library to retrieve paths from the library (this is not a separate
node; just a python library that it calls), and the PathTools python library
which calls the collision_checker service and advertises a topic for displaying
stuff in RViz.
"""
import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import roslib
import rospy
import actionlib
import threading

from tools import NeuralPathTools# import PlanTrajectoryWrapper, InvalidSectionWrapper, DrawPointsWrapper
from tools import NeuralOMPLPathTools
from pathlib.PathLibrary import *
from lightning.msg import Float64Array, RRAction, RRResult
from lightning.msg import StopPlanning, RRStats
from lightning.msg import PlannerType
from lightning.srv import ManagePathLibrary, ManagePathLibraryResponse
from std_msgs.msg import UInt8, Int32, Float32
import sys
import pickle
import time
from architecture.GEM_end2end_model import End2EndMPNet
import numpy as np
import torch.multiprocessing as mp
# Name of this node.
RR_NODE_NAME = "rr_node"
# Name to use for stopping the repair planner. Published from this node.
STOP_PLANNER_NAME = "stop_rr_planning"
# Topic to subscribe to for stopping the whole node in the middle of processing.
STOP_RR_NAME = "stop_all_rr"
# Name of library managing service run from this node.
MANAGE_LIBRARY = "manage_path_library"
PLANNING_SCENE_SERV_NAME = "/get_planning_scene";

STATE_RETRIEVE, STATE_REPAIR, STATE_RETURN_PATH, STATE_FINISHED, STATE_FINISHED = (0, 1, 2, 3, 4)

class RRNode:
    def __init__(self):
        # depending on the argument framework_type, use different classes and settings
        framework_type = rospy.get_param('framework_type')
        if framework_type == 'ompl':
            PlanTrajectoryWrapper = NeuralOMPLPathTools.PlanTrajectoryWrapper
            InvalidSectionWrapper = NeuralOMPLPathTools.InvalidSectionWrapper
            DrawPointsWrapper = NeuralOMPLPathTools.DrawPointsWrapper
        elif framework_type == 'moveit':
            # Make sure that the moveit server is ready before starting up
            rospy.wait_for_service(PLANNING_SCENE_SERV_NAME);
            PlanTrajectoryWrapper = NeuralPathTools.PlanTrajectoryWrapper
            InvalidSectionWrapper = NeuralPathTools.InvalidSectionWrapper
            DrawPointsWrapper = NeuralPathTools.DrawPointsWrapper

        # Retrieve ROS parameters and configuration and cosntruct various objects.
        self.robot_name = rospy.get_param("robot_name")
        self.planner_config_name = rospy.get_param("planner_config_name")
        self.current_joint_names = []
        self.current_group_name = ""
        self.plan_trajectory_wrapper = PlanTrajectoryWrapper("rr", int(rospy.get_param("~num_rr_planners")), \
                                            device_name=rospy.get_param('model/rr_device'))
        self.plan_trajectory_wrapper.neural_planners[0].share_memory()

        self.invalid_section_wrapper = InvalidSectionWrapper()
        self.path_library = PathLibrary(rospy.get_param("~path_library_dir"), rospy.get_param("step_size"), node_size=int(rospy.get_param("~path_library_path_node_size")), sg_node_size=int(rospy.get_param("~path_library_sg_node_size")), dtw_dist=float(rospy.get_param("~dtw_distance")))
        self.num_paths_checked = int(rospy.get_param("~num_paths_to_collision_check"))
        self.stop_lock = threading.Lock()
        self.stop = True
        self.rr_server = actionlib.SimpleActionServer(RR_NODE_NAME, RRAction, execute_cb=self._retrieve_repair, auto_start=False)
        self.rr_server.start()
        self.stop_rr_subscriber = rospy.Subscriber(STOP_RR_NAME, StopPlanning, self._stop_rr_planner)
        self.stop_rr_planner_publisher = rospy.Publisher(STOP_PLANNER_NAME, StopPlanning, queue_size=10)
        self.manage_library_service = rospy.Service(MANAGE_LIBRARY, ManagePathLibrary, self._do_manage_action)
        self.stats_pub = rospy.Publisher("rr_stats", RRStats, queue_size=10)
        self.repaired_sections_lock = threading.Lock()
        self.repaired_sections = []
        self.working_lock = threading.Lock() #to ensure that node is not doing RR and doing a library management action at the same time

        #if draw_points is True, then display points in rviz
        self.draw_points = rospy.get_param("draw_points")
        self.DrawPointsWrapper = DrawPointsWrapper
        if self.draw_points:
            self.draw_points_wrapper = DrawPointsWrapper()
        self._call_classic_planner_res = [None, None]
        self._call_neural_planner_res = [None, None]
        # record the obstacle index for updating the library
        self.obs_i = -1
    def _set_repaired_section(self, index, section):
        """
          After you have done the path planning to repair a section, store
            the repaired path section.

          Args:
            index (int): the index corresponding to the section being repaired.
            section (path, list of list of float): A path to store.
        """
        self.repaired_sections_lock.acquire()
        self.repaired_sections[index] = section
        self.repaired_sections_lock.release()

    def _call_classic_planner(self, start, goal, planning_time):
        """
          Calls a standard planner to plan between two points with an allowed
            planning time.
          Args:
            start (list of float): A joint configuration corresponding to the
              start position of the path.
            goal (list of float): The jount configuration corresponding to the
              goal position for the path.

          Returns:
            path: A list of joint configurations corresponding to the planned
              path.
        """
        rospy.loginfo('RR_action_server: Starting classic planning...')
        ret = None
        classic_planner_time = np.inf
        planner_number = self.plan_trajectory_wrapper.acquire_planner()
        if not self._need_to_stop():
            classic_planner_time, ret = self.plan_trajectory_wrapper.plan_trajectory(start, goal, planner_number, \
                        self.current_joint_names, self.current_group_name, planning_time, self.planner_config_name, \
                        plan_type='rr')
        self.plan_trajectory_wrapper.release_planner(planner_number)
        self._call_classic_planner_res = [classic_planner_time, ret]
        rospy.loginfo('RR_action_server: Finished classic planning.')
        # let planner know that the plan is over
        self.plan_trajectory_wrapper.finished = True

    def _call_neural_planner(self, start, goal, planning_time):
        """
          Calls a neural planner to plan between two points with an allowed
            planning time.
          Args:
            start (list of float): A joint configuration corresponding to the
              start position of the path.
            goal (list of float): The jount configuration corresponding to the
              goal position for the path.

          Returns:
            path: A list of joint configurations corresponding to the planned
              path.
        """
        rospy.loginfo('RR_action_server: Starting neural planning...')
        ret = None
        neural_planner_time = np.inf
        planner_number = self.plan_trajectory_wrapper.acquire_neural_planner()
        # get a smaller planning_time with our default threshold
        default_planning_time = 2.
        planning_time = min(planning_time, default_planning_time)
        if not self._need_to_stop():
            neural_planner_time, ret = self.plan_trajectory_wrapper.neural_plan_trajectory(start, goal, planner_number, \
                            self.current_joint_names, self.current_group_name, planning_time, self.planner_config_name, \
                            plan_type='rr')
        self.plan_trajectory_wrapper.release_neural_planner(planner_number)
        self._call_neural_planner_res = [neural_planner_time, ret]
        rospy.loginfo('RR_action_server: Finished neural planning.')
        # let planner know that the plan is over
        self.plan_trajectory_wrapper.finished = True

    def _call_planner(self, start, goal, planning_time):
        """
          Use multi-threading to plan using classical and neural planners. Use the
          result of the first completed planner.
          Currently does not kill the other process/thread if the first one finished.
          # TODO: add early-stopping for later planner.
          Args:
            start (list of float): A joint configuration corresponding to the
              start position of the path.
            goal (list of float): The jount configuration corresponding to the
              goal position for the path.

          Returns:
            path: A list of joint configurations corresponding to the planned
              path.
        """
        self.plan_trajectory_wrapper.finished = False
        threadList = []
        classical_planner = threading.Thread(target=self._call_classic_planner, args=(start, goal, planning_time))
        neural_planner = threading.Thread(target=self._call_neural_planner, args=(start, goal, planning_time))
        threadList = [classical_planner, neural_planner]
        rospy.loginfo('RR_action_server: Starting multi-thread planning...')
        for th in threadList:
            th.start()
        for th in threadList:
            th.join()
        rospy.loginfo('RR_action_server: Finished multi-thread planning.')
        rospy.loginfo('RR_action_server: classical planner time: %fs' % (self._call_classic_planner_res[0]))
        rospy.loginfo('RR_action_server: neural planner time: %fs' % (self._call_neural_planner_res[0]))

        # by comparing the time of the two planners, use the result of the earlier one
        if self._call_neural_planner_res[0] <= self._call_classic_planner_res[0]:
            # use neural planner result
            rospy.loginfo('RR_action_server: Using Neural Planner result.')
            return [PlannerType.NEURAL, self._call_neural_planner_res[1]]
        else:
            rospy.loginfo('RR_action_server: Using Classical Planner result.')
            return [PlannerType.CLASSIC, self._call_classic_planner_res[1]]


    def _repair_thread(self, index, start, goal, start_index, goal_index, planning_time):
        """
          Handles repairing a portion of the path.
          All that this function really does is to plan from scratch between
            the start and goal configurations and then store the planned path
            in the appropriate places and draws either the repaired path or, if
            the repair fails, the start and goal.

          Args:
            index (int): The index to pass to _set_repaired_section(),
              corresponding to which of the invalid sections of the path we are
              repairing.
            start (list of float): The start joint configuration to use.
            goal (list of float): The goal joint configuration to use.
            start_index (int): The index in the overall path corresponding to
              start. Only used for debugging info.
            goal_index (int): The index in the overall path corresponding to
              goal. Only used for debugging info.
            planning_time (float): Maximum allowed time to spend planning, in
              seconds.
        """
        planner_type, repaired_path = self._call_planner(start, goal, planning_time)
        if self.draw_points:
            if repaired_path is not None and len(repaired_path) > 0:
                rospy.loginfo("RR action server: got repaired section with start = %s, goal = %s" % (repaired_path[0], repaired_path[-1]))
                self.draw_points_wrapper.draw_points(repaired_path, self.current_group_name, "repaired"+str(start_index)+"_"+str(goal_index), self.DrawPointsWrapper.ANGLES, self.DrawPointsWrapper.GREENBLUE, 1.0, 0.01)
        else:
            if self.draw_points:
                rospy.loginfo("RR action server: path repair for section (%i, %i) failed, start = %s, goal = %s" % (start_index, goal_index, start, goal))
                self.draw_points_wrapper.draw_points([start, goal], self.current_group_name, "failed_repair"+str(start_index)+"_"+str(goal_index), self.DrawPointsWrapper.ANGLES, self.DrawPointsWrapper.GREENBLUE, 1.0)
        if self._need_to_stop():
            self._set_repaired_section(index, None)
        else:
            self._set_repaired_section(index, repaired_path)

    def _need_to_stop(self):
        self.stop_lock.acquire();
        ret = self.stop;
        self.stop_lock.release();
        return ret;

    def _set_stop_value(self, val):
        self.stop_lock.acquire();
        self.stop = val;
        self.stop_lock.release();

    def do_retrieved_path_drawing(self, projected, retrieved, invalid):
        """
          Draws the points from the various paths involved in the planning
            in different colors in different namespaces.
          All of the arguments are lists of joint configurations, where each
            joint configuration is a list of joint angles.
          The only distinction between the different arguments being passed in
            are which color the points in question are being drawn in.
          Uses the DrawPointsWrapper to draw the points.

          Args:
            projected (list of list of float): List of points to draw as
              projected between the library path and the actual start/goal
              position. Will be drawn in blue.
            retrieved (list of list of float): The path retrieved straight
              from the path library. Will be drawn in white.
            invalid (list of list of float): List of points which were invalid.
              Will be drawn in red.
        """
        if len(projected) > 0:
            if self.draw_points:
                self.draw_points_wrapper.draw_points(retrieved, self.current_group_name, "retrieved", self.DrawPointsWrapper.ANGLES, self.DrawPointsWrapper.WHITE, 0.1)
                projectionDisplay = projected[:projected.index(retrieved[0])]+projected[projected.index(retrieved[-1])+1:]
                self.draw_points_wrapper.draw_points(projectionDisplay, self.current_group_name, "projection", self.DrawPointsWrapper.ANGLES, self.DrawPointsWrapper.BLUE, 0.2)
                invalidDisplay = []
                for invSec in invalid:
                    invalidDisplay += projected[invSec[0]+1:invSec[-1]]
                self.draw_points_wrapper.draw_points(invalidDisplay, self.current_group_name, "invalid", self.DrawPointsWrapper.ANGLES, self.DrawPointsWrapper.RED, 0.2)

    def _retrieve_repair(self, action_goal):
        """
          Callback which performs the full Retrieve and Repair for the path.
        """
        self.working_lock.acquire()
        self.start_time = time.time()
        # obtain the obstacle id
        obs_i = rospy.wait_for_message('obstacles/obs_i', Int32)
        obs_i = obs_i.data
        # if obstacle id is a new id, then load a new library
        if obs_i != self.obs_i:
            # new id
            self.obs_i = obs_i
            # new robot name: [robot_name]_[obs_id]
            # first try loading the existing library, if not found, then create a new one
            res = self.path_library._load_library(self.robot_name+'_%d' % (obs_i), self.current_joint_names)
            if res == False:
                self.path_library._create_and_load_new_library(self.robot_name+'_%d' % (obs_i), self.current_joint_names)
        # otherwise continue using current library
        self.stats_msg = RRStats()
        self._set_stop_value(False)
        if self.draw_points:
            self.draw_points_wrapper.clear_points()
        rospy.loginfo("RR action server: RR got an action goal")
        s, g = action_goal.start, action_goal.goal
        res = RRResult()
        res.status.status = res.status.FAILURE
        self.current_joint_names = action_goal.joint_names
        self.current_group_name = action_goal.group_name
        projected, retrieved, invalid = [], [], []
        repair_state = STATE_RETRIEVE

        self.stats_msg.init_time = time.time() - self.start_time

        # Go through the retrieve, repair, and return stages of the planning.
        # The while loop should only ever go through 3 iterations, one for each
        #   stage.
        while not self._need_to_stop() and repair_state != STATE_FINISHED:
            if repair_state == STATE_RETRIEVE:
                start_retrieve = time.time()
                projected, retrieved, invalid, retrieved_planner_type = self.path_library.retrieve_path(s, g, self.num_paths_checked, self.robot_name, self.current_group_name, self.current_joint_names)
                self.stats_msg.retrieve_time.append(time.time() - start_retrieve)
                if len(projected) == 0:
                    rospy.loginfo("RR action server: got an empty path for retrieve state")
                    repair_state = STATE_FINISHED
                else:
                    start_draw = time.time()
                    if self.draw_points:
                        self.do_retrieved_path_drawing(projected, retrieved, invalid)
                    self.stats_msg.draw_time.append(time.time() - start_draw)
                    repair_state = STATE_REPAIR
            elif repair_state == STATE_REPAIR:
                start_repair = time.time()
                #print('RR action server: retrieved path:')
                #print(retrieved)
                #print('RR action server: projected path:')
                #print(projected)
                #print('RR action server: invalid:')
                #print(invalid)
                repaired_planner_type, repaired, total_num_paths, total_num_paths_NN, \
                    total_new_node, total_new_node_NN = \
                    self._path_repair(projected, action_goal.allowed_planning_time.to_sec(), invalid_sections=invalid)
                self.stats_msg.repair_time.append(time.time() - start_repair)
                if repaired is None:
                    rospy.loginfo("RR action server: path repair didn't finish")
                    repair_state = STATE_FINISHED
                else:
                    repair_state = STATE_RETURN_PATH
            elif repair_state == STATE_RETURN_PATH:
                start_return = time.time()
                res.status.status = res.status.SUCCESS
                res.retrieved_path = [Float64Array(p) for p in retrieved]
                res.repaired_path = [Float64Array(p) for p in repaired]
                # if total newly generated nodes are 0, it means the library path is not in collision
                # then we don't need to train on them
                if total_new_node == 0:
                    retrieved_planner_type = PlannerType.NEURAL
                    repaired_planner_type = PlannerType.NEURAL
                res.retrieved_planner_type.planner_type = retrieved_planner_type
                res.repaired_planner_type.planner_type = repaired_planner_type
                # added more information of the planner
                res.total_num_paths = total_num_paths
                res.total_num_paths_NN = total_num_paths_NN
                res.total_new_node = total_new_node
                res.total_new_node_NN = total_new_node_NN
                rospy.loginfo('RR action server: total_new_node: %d' % (total_new_node))
                rospy.loginfo('RR action server: total_new_node_NN: %d' % (total_new_node_NN))
                rospy.loginfo("RR action server: total_num_paths_NN before returning: %d" % (total_num_paths_NN))
                rospy.loginfo("RR action server: returning a path")
                repair_state = STATE_FINISHED
                self.stats_msg.return_time = time.time() - start_return
        if repair_state == STATE_RETRIEVE:
            rospy.loginfo("RR action server: stopped before it retrieved a path")
        elif repair_state == STATE_REPAIR:
            rospy.loginfo("RR action server: stopped before it could repair a retrieved path")
        elif repair_state == STATE_RETURN_PATH:
            rospy.loginfo("RR action server: stopped before it could return a repaired path")
        self.rr_server.set_succeeded(res)
        self.stats_msg.total_time = time.time() - self.start_time
        self.stats_pub.publish(self.stats_msg)
        self.working_lock.release()

    def _path_repair(self, original_path, planning_time, invalid_sections=None, use_parallel_repairing=False):
        """
          Goes through each invalid section in a path and calls a planner to
            repair it, with the potential for multi-threading. Returns the
            repaired path.

          Args:
            original_path (path): The original path which needs repairing.
            planning_time (float): The maximum allowed planning time for
              each repair, in seconds.
            invalid_sections (list of pairs of indicies): The pairs of indicies
              describing the invalid sections. If None, then the invalid
              sections will be computed by this function.
            use_parallel_repairing (bool): Whether or not to use multi-threading.

          Returns:
            repaired_planner_type: the planner_type used to repair. (if no need to repair, don't care)
            path: The repaired path.
        """
        zeros_tuple = tuple([0 for i in xrange(len(self.current_joint_names))])
        rospy.loginfo("RR action server: got path with %d points" % len(original_path))

        if invalid_sections is None:
            invalid_sections = self.invalid_section_wrapper.getInvalidSectionsForPath(original_path, self.current_group_name)
        rospy.loginfo("RR action server: invalid sections: %s" % (str(invalid_sections)))
        repaired_planner_type = PlannerType.NEURAL
        total_num_paths = 0
        total_num_paths_NN = 0
        total_new_node = 0
        total_new_node_NN = 0
        if len(invalid_sections) > 0:
            if invalid_sections[0][0] == -1:
                rospy.loginfo("RR action server: Start is not a valid state...nothing can be done")
                return None, None, None, None, None, None
            if invalid_sections[-1][1] == len(original_path):
                rospy.loginfo("RR action server: Goal is not a valid state...nothing can be done")
                return None, None, None, None, None, None

            if use_parallel_repairing:
                #multi-threaded repairing
                # currently this is not working because we can't use Pytorch in multithread or multiprocessing
                self.repaired_sections = [None for i in xrange(len(invalid_sections))]
                #each thread replans an invalid section
                threadList = []
                for i, sec in enumerate(invalid_sections):
                    th = mp.Process(target=self._repair_thread, args=(i, original_path[sec[0]], original_path[sec[-1]], sec[0], sec[-1], planning_time))
                    #th = threading.Thread(target=self._repair_thread, args=(i, original_path[sec[0]], original_path[sec[-1]], sec[0], sec[-1], planning_time))
                    threadList.append(th)
                    th.start()
                for th in threadList:
                    th.join()
                #once all threads return, then the repaired sections can be combined
                for item in self.repaired_sections:
                    if item is None:
                        rospy.loginfo("RR action server: RR node was stopped during repair or repair failed")
                        return None, None, None, None, None, None
                #replace invalid sections with replanned sections
                new_path = original_path[0:invalid_sections[0][0]]
                for i in xrange(len(invalid_sections)):
                    new_path += self.repaired_sections[i]
                    if i+1 < len(invalid_sections):
                        new_path += original_path[invalid_sections[i][1]+1:invalid_sections[i+1][0]]
                new_path += original_path[invalid_sections[-1][1]+1:]
                self.repaired_sections = [] #reset repaired_sections
            else:
                #single-threaded repairing
                rospy.loginfo("RR action server: Got invalid sections: %s" % str(invalid_sections))
                new_path = original_path[0:invalid_sections[0][0]]
                # if at least one classical planner, then set to classical planner
                repaired_planner_type = PlannerType.NEURAL
                total_num_paths = len(invalid_sections)
                total_num_paths_NN = 0
                total_new_node = 0
                total_new_node_NN = 0
                #new_planning_time = planning_time / len(invalid_sections)  # averagely split the planning time
                for i in xrange(len(invalid_sections)):
                    if not self._need_to_stop():
                        #start_invalid and end_invalid must correspond to valid states when passed to the planner
                        start_invalid, end_invalid = invalid_sections[i]
                        rospy.loginfo("RR action server: Requesting path to replace from %d to %d" % (start_invalid, end_invalid))
                        planner_type, repairedSection = self._call_planner(original_path[start_invalid], original_path[end_invalid], planning_time)
                        rospy.loginfo('RR action server: result planner type: %d' % (planner_type))
                        if planner_type == PlannerType.CLASSIC:
                            repaired_planner_type = PlannerType.CLASSIC
                            if repairedSection is not None:
                                total_new_node += len(repairedSection)
                        else:
                            total_num_paths_NN += 1
                            if repairedSection is not None:
                                total_new_node += len(repairedSection)
                                total_new_node_NN += len(repairedSection)
                        rospy.loginfo('RR action server: total_num_paths_NN: %d' % (total_num_paths_NN))
                        ## TODO: returning only invalid sections that are planned by classical method, and train only on them
                        ## TODO: modify library path format to add planner type, so we can train model according to it
                        if repairedSection is None:
                            rospy.loginfo("RR action server: RR section repair was stopped or failed")
                            return None, None, None, None, None, None
                        rospy.loginfo("RR action server: Planner returned a trajectory of %d points for %d to %d" % (len(repairedSection), start_invalid, end_invalid))
                        new_path += repairedSection
                        if i+1 < len(invalid_sections):
                            new_path += original_path[end_invalid+1:invalid_sections[i+1][0]]
                    else:
                        rospy.loginfo("RR action server: RR was stopped while it was repairing the retrieved path")
                        return None, None, None, None, None, None
                new_path += original_path[invalid_sections[-1][1]+1:]
            rospy.loginfo("RR action server: Trajectory after replan has %d points" % len(new_path))
        else:
            new_path = original_path

        rospy.loginfo("RR action server: new trajectory has %i points" % (len(new_path)))
        rospy.loginfo('RR action server: returned total_num_paths_NN: %d' % (total_num_paths_NN))
        return repaired_planner_type, new_path, total_num_paths, total_num_paths_NN, total_new_node, total_new_node_NN

    def _stop_rr_planner(self, msg):
        self._set_stop_value(True)
        rospy.loginfo("RR action server: RR node got a stop message")
        self.stop_rr_planner_publisher.publish(msg)

    def _do_manage_action(self, request):
        """
          Processes a ManagePathLibraryRequest as part of the ManagePathLibrary
            service. Basically, either stores a path in the library or deletes it.
        """
        response = ManagePathLibraryResponse()
        response.result = response.FAILURE
        if request.robot_name == "" or len(request.joint_names) == 0:
            rospy.logerr("RR action server: robot name or joint names were not provided")
            return response

        self.working_lock.acquire()
        if request.action == request.ACTION_STORE:
            rospy.loginfo("RR action server: got a path to store in path library")
            if len(request.path_to_store) > 0:
                new_path = [p.positions for p in request.path_to_store]
                new_path_planner_type = request.planner_type

                if len(request.retrieved_path) == 0:
                    #PFS won so just store the path
                    store_path_result = self.path_library.store_path(new_path, new_path_planner_type, request.robot_name, request.joint_names)
                else:
                    store_path_result = self.path_library.store_path(new_path, new_path_planner_type, request.robot_name, request.joint_names, [p.positions for p in request.retrieved_path])
                response.result = response.SUCCESS
                response.path_stored, response.num_library_paths = store_path_result
            else:
                response.message = "Path to store had no points"
        elif request.action == request.ACTION_DELETE_PATH:
            rospy.loginfo("RR action server: got a request to delete path %i in the path library" % (request.delete_id))
            if self.path_library.delete_path_by_id(request.delete_id, request.robot_name, request.joint_names):
                response.result = response.SUCCESS
            else:
                response.message = "No path in the library had id %i" % (request.delete_id)
        elif request.action == request.ACTION_DELETE_LIBRARY:
            rospy.loginfo("RR action server: got a request to delete library corresponding to robot %s and joints %s" % (request.robot_name, request.joint_names))
            if self.path_library.delete_library(request.robot_name, request.joint_names):
                response.result = response.SUCCESS
            else:
                response.message = "No library corresponding to robot %s and joint names %s exists"
        else:
            rospy.logerr("RR action server: manage path library request did not have a valid action set")
        self.working_lock.release()
        return response

if __name__ == "__main__":
    try:
        rospy.init_node("rr_node")
        RRNode()
        rospy.loginfo("Retrieve-repair: ready")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
