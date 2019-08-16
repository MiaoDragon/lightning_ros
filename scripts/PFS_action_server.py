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

import roslib
import rospy
import actionlib
import threading

from lightning.msg import PFSAction, PFSResult
from lightning.msg import StopPlanning, Float64Array

import tools.PathTools
import tools.OMPLPathTools
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
            PlanTrajectoryWrapper = tools.OMPLPathTools.PlanTrajectoryWrapper
        elif framework_type == 'moveit':
            # Make sure that the moveit server is ready before starting up
            rospy.wait_for_service(PLANNING_SCENE_SERV_NAME);
            PlanTrajectoryWrapper = tools.PathTools.PlanTrajectoryWrapper

        self.plan_trajectory_wrapper = PlanTrajectoryWrapper("pfs")
        self.planner_config_name = rospy.get_param("planner_config_name")
        self.stop_lock = threading.Lock()
        self.current_joint_names = []
        self.current_group_name = ""
        self._set_stop_value(True)
        self.pfs_server = actionlib.SimpleActionServer(PFS_NODE_NAME, PFSAction, execute_cb=self._get_path, auto_start=False)
        self.pfs_server.start()
        self.stop_pfs_subscriber = rospy.Subscriber(STOP_PFS_NAME, StopPlanning, self._stop_pfs_planner)
        self.stop_pfs_planner_publisher = rospy.Publisher(STOP_PLANNER_NAME, StopPlanning, queue_size=10)

    def _get_stop_value(self):
        self.stop_lock.acquire()
        ret = self.stop
        self.stop_lock.release()
        return ret

    def _set_stop_value(self, val):
        self.stop_lock.acquire()
        self.stop = val
        self.stop_lock.release()

    def _call_planner(self, start, goal, planning_time):
        rospy.loginfo("PFS action server: acquiring planner")
        planner_number = self.plan_trajectory_wrapper.acquire_planner()
        rospy.loginfo("PFS action server: got a planner")
        ret = self.plan_trajectory_wrapper.plan_trajectory(start, goal, planner_number, self.current_joint_names, self.current_group_name, planning_time, self.planner_config_name)
        self.plan_trajectory_wrapper.release_planner(planner_number)
        rospy.loginfo("PFS action server: releasing planner")
        return ret

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
            unfiltered = self._call_planner(s, g, action_goal.allowed_planning_time.to_sec())
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
