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
This file contains several wrapper classes for planning paths, performing
  collision checking, retrieving paths, and drawing points in RViz.
None of these classes really perform actualy work but rather are just calling
  ROS services.
"""

import roslib
import rospy

import threading
import sys

from lightning.msg import Float64Array, Float64Array2D, DrawPoints
from lightning.srv import CollisionCheck, CollisionCheckRequest, PathShortcut, PathShortcutRequest
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.msg import JointConstraint, Constraints
import PathTools
# Names of Topics/Services to be advertised/used by these wrappers.
# The name of the collision checking service.
COLLISION_CHECK = "collision_check"
# The name of the path shortcutting service.
SHORTCUT_PATH_NAME = "shortcut_path"
# Topic to publish to for drawing points in RViz.
DISPLAY_POINTS = "draw_points"
# Name of the planner_stoppable services advertised from various lightning nodes.
PLANNER_NAME = "plan_kinematic_path"

class PlanTrajectoryWrapper(PathTools.PlanTrajectoryWrapper):
    """
        This class uses functionalities of PlanTrajectoryWrapper, but overrides the
        planning functions to directly use OMPL planning library.
    """

    def __init__(self, node_type, num_planners=1):
        """
          Constructor for OMPLPlanTrajectoryWrapper.

          Args:
            node_type (string): The type of planner that this is being used by,
              generally "pfs" or "rr".
            num_planners (int): The number of planner nodes that are being used.
        """
        PathTools.PlanTrajectoryWrapper.__init__(self, node_type, num_planners)
        ## add OMPL setting for different environments
        self.env_name = rospy.get_param('env_name')
        if self.env_name == 's2d':
            data_loader = data_loader_2d
            IsInCollision = plan_s2d.IsInCollision
            # create an SE2 state space
            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
            time_limit = 20.
        elif self.env_name == 'c2d':
            data_loader = data_loader_2d
            IsInCollision = plan_c2d.IsInCollision
            # create an SE2 state space
            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
            time_limit = 10.
        elif self.env_name == 'r2d':
            data_loader = data_loader_r2d
            IsInCollision = plan_r2d.IsInCollision
            # create an SE2 state space
            space = ob.SE2StateSpace()
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
            time_limit = 50.
        elif self.env_name == 'r3d':
            data_loader = data_loader_r3d
            IsInCollision = plan_r3d.IsInCollision
            # create an SE2 state space
            space = ob.RealVectorStateSpace(3)
            bounds = ob.RealVectorBounds(3)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
            time_limit = 20.
    def plan_trajectory(self, start_point, goal_point, planner_number, joint_names, group_name, planning_time, planner_config_name):
        """
            Use OMPL library for planning.
        """
        pass
class ShortcutPathWrapper(PathTools.ShortcutPathWrapper):
    """
      This is a very thin wrapper over the path shortcutting service.
    """

    def shortcut_path(self, original_path, group_name):
        """
          Shortcuts a path, where the path is for a given group name.

          Args:
            original_path (list of list of float): The path, represented by
              a list of individual joint configurations.
            group_name (str): The group for which the path was created.

          Return:
            list of list of float: The shortcutted version of the path.
        """
        shortcut_path_client = rospy.ServiceProxy(SHORTCUT_PATH_NAME, PathShortcut)
        shortcut_req = PathShortcutRequest()
        shortcut_req.path = [Float64Array(p) for p in original_path]
        shortcut_req.group_name = group_name
        rospy.wait_for_service(SHORTCUT_PATH_NAME)
        response = shortcut_path_client(shortcut_req)
        return [p.values for p in response.new_path]

class InvalidSectionWrapper(PathTools.InvalidSectionWrapper):
    """
      This is a very thin wrapper over the collision checking service.
    """

    def get_invalid_sections_for_path(self, original_path, group_name):
        """
          Returns the invalid sections for a single path.

          Args:
            original_path (list of list of float): The path to collision check,
              represnted by a list of individual joint configurations.
            group_name (str): The joint group for which the path was created.

          Return:
            list of pairs of indicies, where each index in a pair is the start
              and end of an invalid section.
        """
        section = self.get_invalid_sections_for_paths([original_path], group_name)
        if len(section) > 0:
            return section[0]
        else:
            return None

    def get_invalid_sections_for_paths(self, orig_paths, group_name):
        """
          Returns the invalid sections for a set of paths.

          Args:
            orig_paths (list of paths): The paths to collision check,
              represnted by a list of individual joint configurations.
            group_name (str): The joint group for which the paths were created.

          Return:
            list of list of pairs of indicies, where each index in a pair is the
              start and end of an invalid section.
        """
        collision_check_client = rospy.ServiceProxy(COLLISION_CHECK, CollisionCheck)
        cc_req = CollisionCheckRequest();
        cc_req.paths = [Float64Array2D([Float64Array(point) for point in path]) for path in orig_paths];
        cc_req.group_name = group_name
        rospy.loginfo("Plan Trajectory Wrapper: sending request to collision checker")
        rospy.wait_for_service(COLLISION_CHECK)
        response = collision_check_client(cc_req);
        return [[sec.values for sec in individualPathSections.points] for individualPathSections in response.invalid_sections];

class DrawPointsWrapper(PathTools.DrawPointsWrapper):
    pass

if __name__ == "__main__":
    if len(sys.argv) == 8:
        isw = InvalidSectionWrapper()
        path = [float(sys.argv[i]) for i in xrange(1, len(sys.argv))]
        print isw.get_invalid_sections_for_path([path])
