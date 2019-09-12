import numpy as np
import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
# Service for checking if individual state is valid
STATE_VALID = 'check_state_validity'
DEFAULT_SV_SERVICE = "/check_state_validity"

def stateValidate(robot_state, group_name, constraints=None, print_depth=False):
    sv_srv = rospy.ServiceProxy(DEFAULT_SV_SERVICE, GetStateValidity)
    rospy.loginfo("Collision Checking by using check_state_validity service...")
    #rospy.wait_for_service(STATE_VALID)
    gsvr = GetStateValidityRequest()
    gsvr.robot_state = robot_state
    gsvr.group_name = group_name
    if constraints != None:
        gsvr.constraints = constraints
    result = sv_srv.call(gsvr)

    if (not result.valid):
        contact_depths = []
        for i in xrange(len(result.contacts)):
            contact_depths.append(result.contacts[i].depth)

        max_depth = max(contact_depths)
        if max_depth < 0.0001:
            return True
        else:
            return False

    return result.valid



def IsInCollision(x, filler_robot_state, joint_name):
    robot_state = RobotState()
    filler_robot_state[10:17] = x
    robot_state.joint_state.name = joint_name
    robot_state.joint_state.position = tuple(filler_robot_state)
    collision_free = stateValidate(robot_state, group_name="right_arm")
    return (not collision_free)
