/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/** \author Sachin Chitta, Ioan Sucan */

#include <lightning/planner_stoppable/ompl_ros_stoppable.h>
#include <sstream>

/* THIS CLASS IS VERY SIMILAR TO OMPL_ROS. THE DIFFERENCE IS THAT THIS CLASS USES A PLANNER THAT CAN BE TERMINATED. */

OmplRosStoppable::OmplRosStoppable(void): private_nh_("~")
{
    collision_models_interface_ = new planning_environment::CollisionModelsInterface("robot_description");
    ROS_INFO("OmplRosStoppable started");
}

//addition
OmplRosStoppable::OmplRosStoppable(std::string name, std::string stop_name): private_nh_("~")
{
    planner_name_ = name;
    stop_name_ = stop_name;
    collision_models_interface_ = new planning_environment::CollisionModelsInterface("robot_description");
    ROS_INFO("OmplRosStoppable started");
}

/** Free the memory */
OmplRosStoppable::~OmplRosStoppable(void)
{
    delete collision_models_interface_;
}

void OmplRosStoppable::run(void)
{
    if(!initialize(private_nh_.getNamespace()))
        return;
    if (collision_models_interface_->loadedModels())
    {
        plan_path_service_ = private_nh_.advertiseService(planner_name_, &OmplRosStoppable::computePlan, this);

        //addition
        stop_planning_subscriber_ = nh_.subscribe(stop_name_, 10, &OmplRosStoppable::stop_planning, this);

        //addition
        ROS_INFO("Listening on %s and %s for ompl ros interface", planner_name_.c_str(), stop_name_.c_str());

        private_nh_.param<bool>("publish_diagnostics", publish_diagnostics_,false);
        if(publish_diagnostics_)
            diagnostic_publisher_ = private_nh_.advertise<ompl_ros_interface::OmplPlannerDiagnostics>("diagnostics", 1);
    }
    else
        ROS_ERROR("Collision models not loaded.");
}

bool OmplRosStoppable::initialize(const std::string &param_server_prefix)
{
    std::vector<std::string> group_names;
    if(!getGroupNamesFromParamServer(param_server_prefix,group_names))
    {
        ROS_ERROR("Could not find groups for planning under %s",param_server_prefix.c_str());
        return false;
    }
    if(!initializePlanningMap(param_server_prefix,group_names))
    {
        ROS_ERROR("Could not initialize planning groups from the param server");
        return false;
    }

    if(!private_nh_.hasParam("default_planner_config"))
    {
        ROS_ERROR("No default planner configuration defined under 'default_planner_config'. A default planner must be defined from among the configured planners");
        return false;
    }

    private_nh_.param<std::string>("default_planner_config",default_planner_config_,"SBLkConfig1");
    for(unsigned int i=0; i < group_names.size(); i++)
    {
        std::string location = default_planner_config_ + "[" + group_names[i] + "]";
        if(planner_map_.find(location) == planner_map_.end())
        {
            ROS_ERROR("The default planner configuration %s has not been defined for group %s. The default planner must be configured for every group in your ompl_planning.yaml file", default_planner_config_.c_str(), group_names[i].c_str());
            return false;
        }
    }

    return true;
};

bool OmplRosStoppable::getGroupNamesFromParamServer(const std::string &param_server_prefix, 
        std::vector<std::string> &group_names)
{
    XmlRpc::XmlRpcValue group_list;
    if(!private_nh_.getParam(param_server_prefix+"/groups", group_list))
    {
        ROS_ERROR("Could not find parameter %s on param server",(param_server_prefix+"/groups").c_str());
        return false;
    }
    if(group_list.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
        ROS_ERROR("Group list should be of XmlRpc Array type");
        return false;
    } 
    for (int32_t i = 0; i < group_list.size(); ++i) 
    {
        if(group_list[i].getType() != XmlRpc::XmlRpcValue::TypeString)
        {
            ROS_ERROR("Group names should be strings");
            return false;
        }
        group_names.push_back(static_cast<std::string>(group_list[i]));
        ROS_DEBUG("Adding group: %s",group_names.back().c_str());
    }
    return true;
};

bool OmplRosStoppable::initializePlanningMap(const std::string &param_server_prefix,
        const std::vector<std::string> &group_names)
{
    for(unsigned int i=0; i < group_names.size(); i++)
    {
        XmlRpc::XmlRpcValue planner_list;
        if(!private_nh_.getParam(param_server_prefix+"/"+group_names[i]+"/planner_configs", planner_list))
        {
            ROS_ERROR("Could not find parameter %s on param server",(param_server_prefix+"/"+group_names[i]+"/planner_configs").c_str());
            return false;
        }
        ROS_ASSERT(planner_list.getType() == XmlRpc::XmlRpcValue::TypeArray);    
        for (int32_t j = 0; j < planner_list.size(); ++j) 
        {
            if(planner_list[j].getType() != XmlRpc::XmlRpcValue::TypeString)
            {
                ROS_ERROR("Planner names must be of type string");
                return false;
            }
            std::string planner_config = static_cast<std::string>(planner_list[j]);
            if(!initializePlanningInstance(param_server_prefix,group_names[i],planner_config))
            {
                ROS_ERROR("Could not add planner for group %s and planner_config %s",group_names[i].c_str(),planner_config.c_str());
                return false;
            }
            else
            {
                ROS_DEBUG("Adding planning group config: %s",(planner_config+"["+group_names[i]+"]").c_str());
            }
        }    
    } 
    return true;
};

bool OmplRosStoppable::initializePlanningInstance(const std::string &param_server_prefix,
        const std::string &group_name,
        const std::string &planner_config_name)
{
    std::string location = planner_config_name+"["+group_name+"]";
    if (planner_map_.find(location) != planner_map_.end())
    {
        ROS_WARN("Re-definition of '%s'", location.c_str());
        return true;
    }

    if(!private_nh_.hasParam(param_server_prefix+"/"+group_name+"/planner_type"))
    {
        ROS_ERROR_STREAM("Planner type not defined for group " << group_name << " param name " << param_server_prefix+"/"+group_name+"/planner_type");
        return false;
    }

    std::string planner_type;
    private_nh_.getParam(param_server_prefix+"/"+group_name+"/planner_type",planner_type);
    if(planner_type == "JointPlanner")
    {
        boost::shared_ptr<OmplRosStoppableJointPlanner> new_planner;
        new_planner.reset(new OmplRosStoppableJointPlanner());
        if(!new_planner->initialize(ros::NodeHandle(param_server_prefix),group_name,planner_config_name,collision_models_interface_))
        {
            new_planner.reset();
            ROS_ERROR("Could not configure planner for group %s with config %s",group_name.c_str(),planner_config_name.c_str());
            return false;
        }
        planner_map_[location] = new_planner;
    }
    /*
    else if(planner_type == "RPYIKTaskSpacePlanner")
    {
        boost::shared_ptr<ompl_ros_interface::OmplRosRPYIKTaskSpacePlanner> new_planner;
        new_planner.reset(new ompl_ros_interface::OmplRosRPYIKTaskSpacePlanner());
        if(!new_planner->initialize(ros::NodeHandle(param_server_prefix),group_name,planner_config_name,collision_models_interface_))
        {
            new_planner.reset();
            ROS_ERROR("Could not configure planner for group %s with config %s",group_name.c_str(),planner_config_name.c_str());
            return false;
        }
        planner_map_[location] = new_planner;
    }*/
    else
    {
        ROS_ERROR("No planner type %s available",planner_type.c_str());
        std::string cast;
        private_nh_.getParam(param_server_prefix+"/"+group_name, cast);
        ROS_ERROR_STREAM("Here " << cast);
        return false;
    }

    return true;
};

//addition
void OmplRosStoppable::stop_planning(const lightning::StopPlanning::ConstPtr & msg) {
    std::string location, planner_id;
    ROS_INFO("%s got a stop planning message", planner_name_.c_str());
    if(msg->planner_id == "") {
        planner_id = default_planner_config_;
    } else {
        planner_id = msg->planner_id; 
    }
    location = planner_id + "[" +msg->group_name + "]";
    if(planner_map_.find(location) == planner_map_.end()) {
        ROS_ERROR("Could not find planner at %s", location.c_str());
    } else {
        ROS_DEBUG("Using planner config %s",location.c_str());
        planner_map_[location]->stop_planning();
    }
}        

bool OmplRosStoppable::computePlan(arm_navigation_msgs::GetMotionPlan::Request &request, arm_navigation_msgs::GetMotionPlan::Response &response)
{
    std::string location;
    std::string planner_id;
    if(request.motion_plan_request.planner_id == "")
        planner_id = default_planner_config_;
    else
        planner_id = request.motion_plan_request.planner_id; 
    location = planner_id + "[" +request.motion_plan_request.group_name + "]";
    if(planner_map_.find(location) == planner_map_.end())
    {
        ROS_ERROR("Could not find requested planner %s", location.c_str());
        response.error_code.val = arm_navigation_msgs::ArmNavigationErrorCodes::INVALID_PLANNER_ID;
        return true;
    }
    else
    {
        ROS_DEBUG("Using planner config %s",location.c_str());
    }

    //addition
    std::ostringstream start, goal;
    for (int i = 0; i < (int)request.motion_plan_request.start_state.joint_state.position.size(); i++) {
        start << request.motion_plan_request.start_state.joint_state.position[i] << " ";
    }
    for (int i = 0; i < (int)request.motion_plan_request.goal_constraints.joint_constraints.size(); i++) {
        goal << request.motion_plan_request.goal_constraints.joint_constraints[i].position << " ";
    }
    ROS_INFO("ompl ros stoppable: got a request to plan between %s and %s", start.str().c_str(), goal.str().c_str());
    
    planner_map_[location]->computePlan(request,response);
    if(publish_diagnostics_)
    {
        ompl_ros_interface::OmplPlannerDiagnostics msg;
        if(response.error_code.val != arm_navigation_msgs::ArmNavigationErrorCodes::SUCCESS) {
            msg.summary = "Planning Failed";
            std::string filename = "planning_failure_";
            std::string str = boost::lexical_cast<std::string>(ros::Time::now().toSec());
            filename += str;
            collision_models_interface_->writePlanningSceneBag(filename,
                    collision_models_interface_->getLastPlanningScene());
            collision_models_interface_->appendMotionPlanRequestToPlanningSceneBag(filename,
                    "motion_plan_request",
                    request.motion_plan_request);
        }
        else
            msg.summary = "Planning Succeeded";

        msg.group = request.motion_plan_request.group_name;
        msg.planner = planner_id;
        msg.result =  arm_navigation_msgs::armNavigationErrorCodeToString(response.error_code);
        if(response.error_code.val == arm_navigation_msgs::ArmNavigationErrorCodes::SUCCESS)
        {
            msg.planning_time = response.planning_time.toSec();
            msg.trajectory_size = response.trajectory.joint_trajectory.points.size();
            msg.trajectory_duration = response.trajectory.joint_trajectory.points.back().time_from_start.toSec()-response.trajectory.joint_trajectory.points.front().time_from_start.toSec();    
            //      msg.state_allocator_size = planner_map_[location]->planner_->getSpaceInformation()->getStateAllocator().sizeInUse();
        }
        diagnostic_publisher_.publish(msg);
    }
    return true;
};

boost::shared_ptr<OmplRosStoppablePlanningGroup>& OmplRosStoppable::getPlanner(const std::string &group_name,
        const std::string &planner_config_name)
{
    std::string location = planner_config_name + "[" + group_name + "]";
    if(planner_map_.find(location) == planner_map_.end())
    {
        ROS_ERROR("Could not find requested planner %s", location.c_str());
        return empty_ptr;
    }
    else
    {
        ROS_DEBUG("Using planner config %s",location.c_str());
        return planner_map_[location];
    }
};
