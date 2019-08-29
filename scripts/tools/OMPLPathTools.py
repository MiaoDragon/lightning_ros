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
import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from lightning.msg import Float64Array, Float64Array2D, DrawPoints
from lightning.srv import CollisionCheck, CollisionCheckRequest, PathShortcut, PathShortcutRequest
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.msg import JointConstraint, Constraints
from std_msgs.msg import Float64
import PathTools

from ompl import base as ob
from ompl import geometric as og
from experiments.simple import plan_general
from experiments.simple import plan_c2d, plan_s2d, plan_r2d, plan_r3d
#from experiments.simple import data_loader_2d, data_loader_r2d, data_loader_r3d
from experiments.simple import utility_s2d, utility_c2d, utility_r2d, utility_r3d
import argparse
import pickle
import sys
import time
import os
import numpy as np
import time
# Names of Topics/Services to be advertised/used by these wrappers.
# The name of the collision checking service.
COLLISION_CHECK = "collision_check"
# The name of the path shortcutting service.
SHORTCUT_PATH_NAME = "shortcut_path"
# Topic to publish to for drawing points in RViz.
DISPLAY_POINTS = "draw_points"
# Name of the planner_stoppable services advertised from various lightning nodes.
PLANNER_NAME = "plan_kinematic_path"
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        planner = og.BITstar(si)
        planner.setPruning(False)
        planner.setSamplesPerBatch(200)
        planner.setRewireFactor(20.)
        return planner
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == 'rrtconnect':
        return og.RRTConnect(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def getPathLengthObjective(si, length):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(length))
    return obj

class PlanTrajectoryWrapper(PathTools.PlanTrajectoryWrapper):
    """
        This class uses functionalities of PlanTrajectoryWrapper, but overrides the
        planning functions to directly use OMPL planning library.
    """

    def __init__(self, node_type, num_planners=1):
        """
          Constructor for OMPLPlanTrajectoryWrapper.

          In orignal PathTools, it is using multi-threading with multiple ROS services/machines.
          Hence it required locks to record if each service/machine is idle or busy.
          However, in this version we don't need to worry about that, since the planning function
          does not utilize any ROS services. It is only calling the OMPL library code (if there
          is no service involved in OMPL). And all shared variables won't be in danger
          in the trajectory_planning function. Hence it is safe to call many trajectory planning
          jobs, without the need of using locks.
          However, for API safety and future extension (we may use neural network in this setting)
          we'll still use the locks.
          But it can be removed from this library.

          Args:
            node_type (string): The type of planner that this is being used by,
              generally "pfs" or "rr".
            num_planners (int): The number of planner nodes that are being used.
        """
        PathTools.PlanTrajectoryWrapper.__init__(self, node_type, num_planners)
        ## add OMPL setting for different environments
        self.env_name = rospy.get_param('env_name')
        self.planner_name = rospy.get_param('planner_name')
        if self.env_name == 's2d':
            #data_loader = data_loader_2d
            IsInCollision = plan_s2d.IsInCollision
            # create an SE2 state space
            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
        elif self.env_name == 'c2d':
            #data_loader = data_loader_2d
            IsInCollision = plan_c2d.IsInCollision
            # create an SE2 state space
            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
        elif self.env_name == 'r2d':
            #data_loader = data_loader_r2d
            IsInCollision = plan_r2d.IsInCollision
            # create an SE2 state space
            space = ob.SE2StateSpace()
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
        elif self.env_name == 'r3d':
            #data_loader = data_loader_r3d
            IsInCollision = plan_r3d.IsInCollision
            # create an SE2 state space
            space = ob.RealVectorStateSpace(3)
            bounds = ob.RealVectorBounds(3)
            bounds.setLow(-20)
            bounds.setHigh(20)
            space.setBounds(bounds)
        self.IsInCollision = IsInCollision
        self.space = space
        # for thread-safety, should not modify shared vars
        #self.si = ob.SpaceInformation(space)

    def plan_trajectory(self, start_point, goal_point, planner_number, joint_names, group_name, planning_time, planner_config_name, plan_type='pfs'):
        """
            Use OMPL library for planning. Obtain obstacle information from rostopic for
            collision checking
        """
        # obtain obstacle information through rostopic
        rospy.loginfo("%s Plan Trajectory Wrapper: waiting for obstacle message..." % (rospy.get_name()))
        obc = rospy.wait_for_message('obstacles/obc', Float64Array2D)
        # obs = rospy.wait_for_message('obstacles/obs', Float64Array2D)
        obc = [obc_i.values for obc_i in obc.points]
        obc = np.array(obc)
        rospy.loginfo("%s Plan Trajectory Wrapper: obstacle message received." % (rospy.get_name()))
        # depending on plan type, obtain path_length from published topic or not
        if plan_type == 'pfs':
            # obtain path length through rostopic
            rospy.loginfo("%s Plan Trajectory Wrapper: waiting for planning path length message..." % (rospy.get_name()))
            path_length = rospy.wait_for_message('planning/path_length_threshold', Float64)
            path_length = path_length.data
            rospy.loginfo("%s Plan Trajectory Wrapper: planning path length received." % (rospy.get_name()))
        elif plan_type == 'rr':
            path_length = np.inf  # set a very large path length because we only want feasible paths
        # reshape
        # plan
        IsInCollision = self.IsInCollision
        rospy.loginfo("%s Plan Trajectory Wrapper: start planning..." % (rospy.get_name()))
        # create a simple setup object
        start = ob.State(self.space)
        # we can pick a random start state...
        # ... or set specific values
        for k in xrange(len(start_point)):
            start[k] = start_point[k]
        goal = ob.State(self.space)
        for k in xrange(len(goal_point)):
            goal[k] = goal_point[k]
        def isStateValid(state):
            return not IsInCollision(state, obc)
        si = ob.SpaceInformation(self.space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
        si.setup()
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        pdef.setOptimizationObjective(getPathLengthObjective(si, path_length))

        ss = allocatePlanner(si, self.planner_name)
        ss.setProblemDefinition(pdef)
        ss.setup()
        plan_time = time.time()
        solved = ss.solve(planning_time)
        plan_time = time.time() - plan_time
        if solved:
            rospy.loginfo("%s Plan Trajectory Wrapper: OMPL Planner solved successfully." % (rospy.get_name()))
            # obtain planned path
            ompl_path = pdef.getSolutionPath().getStates()
            solutions = np.zeros((len(ompl_path),2))
            for k in xrange(len(ompl_path)):
                solutions[k][0] = float(ompl_path[k][0])
                solutions[k][1] = float(ompl_path[k][1])
            return plan_time, solutions.tolist()
        else:
            return np.inf, None


class ShortcutPathWrapper(PathTools.ShortcutPathWrapper):
    """
      This is a very thin wrapper over the path shortcutting service.
    """
    def __init__(self):
        ## add OMPL setting for different environments
        self.env_name = rospy.get_param('env_name')
        self.planner = rospy.get_param('planner_name')
        if self.env_name == 's2d':
            #data_loader = data_loader_2d
            IsInCollision = plan_s2d.IsInCollision
            # create an SE2 state space
            time_limit = 20.
        elif self.env_name == 'c2d':
            #data_loader = data_loader_2d
            IsInCollision = plan_c2d.IsInCollision
            # create an SE2 state space
            time_limit = 10.
        elif self.env_name == 'r2d':
            #data_loader = data_loader_r2d
            IsInCollision = plan_r2d.IsInCollision
            # create an SE2 state space
            time_limit = 50.
        elif self.env_name == 'r3d':
            #data_loader = data_loader_r3d
            IsInCollision = plan_r3d.IsInCollision
            # create an SE2 state space
            time_limit = 20.
        self.IsInCollision = IsInCollision

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
        # obtain obstacle information through rostopic
        rospy.loginfo("Shortcut Path Wrapper: waiting for obstacle message...")
        obc = rospy.wait_for_message('obstacles/obc', Float64Array2D)
        # obs = rospy.wait_for_message('obstacles/obs', Float64Array2D)
        obc = [obc_i.values for obc_i in obc.points]
        obc = np.array(obc)
        original_path = np.array(original_path)
        rospy.loginfo("Shortcut Path Wrapper: obstacle message received.")
        path = plan_general.lvc(original_path, obc, self.IsInCollision, step_sz=rospy.get_param("step_size"))
        path = np.array(path).tolist()
        return path

class InvalidSectionWrapper(PathTools.InvalidSectionWrapper):
    """
        This uses our user-defined collision checker
    """
    def __init__(self):
        ## add OMPL setting for different environments
        self.env_name = rospy.get_param('env_name')
        self.planner = rospy.get_param('planner_name')
        if self.env_name == 's2d':
            IsInCollision = plan_s2d.IsInCollision
        elif self.env_name == 'c2d':
            IsInCollision = plan_c2d.IsInCollision
        elif self.env_name == 'r2d':
            IsInCollision = plan_r2d.IsInCollision
        elif self.env_name == 'r3d':
            IsInCollision = plan_r3d.IsInCollision
        self.IsInCollision = IsInCollision
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
        # obtain obstacle information through rostopic
        rospy.loginfo("Invalid Section Wrapper: waiting for obstacle message...")
        obc = rospy.wait_for_message('obstacles/obc', Float64Array2D)
        # obs = rospy.wait_for_message('obstacles/obs', Float64Array2D)
        obc = [obc_i.values for obc_i in obc.points]
        obc = np.array(obc)
        rospy.loginfo("Invalid Section Wrapper: obstacle message received.")
        # transform from orig_paths
        # for each path, check the invalid sections
        inv_sec_paths = []
        for orig_path in orig_paths:
            inv_sec_path = []
            start_i = 0
            end_i = 0
            mode = True # valid till now
            for point_i in range(len(orig_path)):
                point = orig_path[point_i]
                if self.IsInCollision(point, obc):
                    mode = False
                else:
                    if mode:
                        # if valid till now, update start
                        start_i = point_i
                    else:
                        # if not valid, then update end
                        end_i = point_i
                        start_i = point_i
                        inv_sec_path.append([start_i, end_i])
                        mode = True  # become valid again
            inv_sec_paths.append(inv_sec_path)
        return inv_sec_paths

class DrawPointsWrapper(PathTools.DrawPointsWrapper):
    pass

if __name__ == "__main__":
    if len(sys.argv) == 8:
        isw = InvalidSectionWrapper()
        path = [float(sys.argv[i]) for i in xrange(1, len(sys.argv))]
        print isw.get_invalid_sections_for_path([path])
