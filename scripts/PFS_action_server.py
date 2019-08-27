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
(see run_lightning.py) to run the standard planning algorithm (exactly which
planner is run is specified in planners.launch). This is essentially a thin
wrapper over the planner_stoppable type node, which is itself a thin wrapper
over OMPL.
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

from lightning.msg import PFSAction, PFSResult
from lightning.msg import StopPlanning, Float64Array
from lightning.msg import PlannerType
from tools import NeuralPathTools
from tools import NeuralOMPLPathTools
import time
import numpy as np
# The name of this node.
PFS_NODE_NAME = "pfs_node";
# The topic to Publish which tells the actual planners to stop.
STOP_PLANNER_NAME = "stop_pfs_planning";
# The topic to listen to which is used to tell us to stop.
STOP_PFS_NAME = "stop_all_pfs";
# The service which, when up, indicates that moveit is started.
PLANNING_SCENE_SERV_NAME = "/get_planning_scene";

class PFSNode:
    def __init__(self):
        # depending on the argument framework_type, use different classes and settings
        framework_type = rospy.get_param('framework_type')
        if framework_type == 'ompl':
            PlanTrajectoryWrapper = NeuralOMPLPathTools.PlanTrajectoryWrapper
        elif framework_type == 'moveit':
            # Make sure that the moveit server is ready before starting up
            rospy.wait_for_service(PLANNING_SCENE_SERV_NAME);
            PlanTrajectoryWrapper = NeuralPathTools.PlanTrajectoryWrapper

        self.plan_trajectory_wrapper = PlanTrajectoryWrapper("pfs", device_name=rospy.get_param('model/pfs_device'))
        self.plan_trajectory_wrapper.neural_planners[0].share_memory()

        self.planner_config_name = rospy.get_param("planner_config_name")
        self.stop_lock = threading.Lock()
        self.current_joint_names = []
        self.current_group_name = ""
        self._set_stop_value(True)
        self.pfs_server = actionlib.SimpleActionServer(PFS_NODE_NAME, PFSAction, execute_cb=self._get_path, auto_start=False)
        self.pfs_server.start()
        self.stop_pfs_subscriber = rospy.Subscriber(STOP_PFS_NAME, StopPlanning, self._stop_pfs_planner)
        self.stop_pfs_planner_publisher = rospy.Publisher(STOP_PLANNER_NAME, StopPlanning, queue_size=10)
        self._call_classic_planner_res = [None, None]
        self._call_neural_planner_res = [None, None]
    def _get_stop_value(self):
        self.stop_lock.acquire()
        ret = self.stop
        self.stop_lock.release()
        return ret

    def _need_to_stop(self):
        # alias to above function
        self.stop_lock.acquire();
        ret = self.stop;
        self.stop_lock.release();
        return ret;

    def _set_stop_value(self, val):
        self.stop_lock.acquire()
        self.stop = val
        self.stop_lock.release()


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
        rospy.loginfo('PFS_action_server: Starting classic planning...')
        ret = None
        classic_planner_time = np.inf # set infinity so that when stopped, won't use this
        planner_number = self.plan_trajectory_wrapper.acquire_planner()
        if not self._need_to_stop():
            classic_planner_time, ret = self.plan_trajectory_wrapper.plan_trajectory(start, goal, planner_number, self.current_joint_names, self.current_group_name, planning_time, self.planner_config_name)
        self.plan_trajectory_wrapper.release_planner(planner_number)
        self._call_classic_planner_res = [classic_planner_time, ret]
        rospy.loginfo('PFS_action_server: Finished classic planning.')


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
        rospy.loginfo('PFS_action_server: Starting neural planning...')
        ret = None
        neural_planner_time = np.inf
        planner_number = self.plan_trajectory_wrapper.acquire_neural_planner()
        if not self._need_to_stop():
            neural_planner_time, ret = self.plan_trajectory_wrapper.neural_plan_trajectory(start, goal, planner_number, self.current_joint_names, self.current_group_name, planning_time, self.planner_config_name)
        self.plan_trajectory_wrapper.release_neural_planner(planner_number)
        self._call_neural_planner_res = [neural_planner_time, ret]
        rospy.loginfo('PFS_action_server: Finished neural planning.')

    def _call_planner(self, start, goal, planning_time):
        """
          Use multi-threading to plan using classical and neural planners. Use the
          result of the first completed planner.
          Currently does not kill the other process/thread if the first one finished.
          # TODO: 1. add loginfo
                  2. add early-stopping for later planner.
          Args:
            start (list of float): A joint configuration corresponding to the
              start position of the path.
            goal (list of float): The jount configuration corresponding to the
              goal position for the path.

          Returns:
            path: A list of joint configurations corresponding to the planned
              path.
        """
        threadList = []
        classical_planner = threading.Thread(target=self._call_classic_planner, args=(start, goal, planning_time))
        neural_planner = threading.Thread(target=self._call_neural_planner, args=(start, goal, planning_time))
        threadList = [classical_planner, neural_planner]
        rospy.loginfo('PFS_action_server: Starting multi-thread planning...')
        for th in threadList:
            th.start()
        for th in threadList:
            th.join()
        rospy.loginfo('PFS_action_server: Finished multi-thread planning.')
        rospy.loginfo('PFS_action_server: classical planner time: %fs' % (self._call_classic_planner_res[0]))
        rospy.loginfo('PFS_action_server: neural planner time: %fs' % (self._call_neural_planner_res[0]))
        # by comparing the time of the two planners, use the result of the earlier one
        if self._call_neural_planner_res[0] <= self._call_classic_planner_res[0]:
            # use neural planner result
            rospy.loginfo('PFS_action_server: Using Neural Planner result.')
            return [PlannerType.NEURAL, self._call_neural_planner_res[1]]
        else:
            rospy.loginfo('PFS_action_server: Using Classical Planner result.')
            return [PlannerType.CLASSIC, self._call_classic_planner_res[1]]


    def _get_path(self, action_goal):
        """
          Callback which retrieves a path given a goal.
        """
        self._set_stop_value(False)
        rospy.loginfo("PFS action server: PFS got an action goal")
        res = PFSResult()
        s, g = action_goal.start, action_goal.goal
        res.status.status = res.status.FAILURE
        self.current_joint_names = action_goal.joint_names
        self.current_group_name = action_goal.group_name
        if not self._get_stop_value():
            planner_type, unfiltered = self._call_planner(s, g, action_goal.allowed_planning_time.to_sec())
            if unfiltered is None:
                self.pfs_server.set_succeeded(res)
                return
        else:
            rospy.loginfo("PFS action server: PFS was stopped before it started planning")
            self.pfs_server.set_succeeded(res)
            return

        if not self._get_stop_value():
            # The planner succeeded; actually return the path in the result.
            res.status.status = res.status.SUCCESS
            res.path = [Float64Array(p) for p in unfiltered]
            res.planner_type = planner_type
            self.pfs_server.set_succeeded(res)
            return
        else:
            rospy.loginfo("PFS action server: PFS found a path but RR succeeded first")
            self.pfs_server.set_succeeded(res)
            return

    def _stop_pfs_planner(self, msg):
        # Tells code within this class to stop.
        self._set_stop_value(True)
        rospy.loginfo("PFS action server: PFS node got a stop message")

        # Actually stops planning nodes.
        self.stop_pfs_planner_publisher.publish(msg)

if __name__ == "__main__":
    try:
        rospy.init_node("pfs_node")
        PFSNode()
        rospy.loginfo("PFSNode is ready")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass;
