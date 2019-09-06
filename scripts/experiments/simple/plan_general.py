import sys, os
import rospkg
rospack = rospkg.RosPack()
top_path = rospack.get_path('lightning')
sys.path.insert(1, top_path+'/scripts')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import numpy as np
from tools.utility import *
import time
DEFAULT_STEP = 2.
def removeCollision(path, obc, IsInCollision):
    new_path = []
    # rule out nodes that are already in collision
    for i in range(0,len(path)):
        if not IsInCollision(path[i].numpy(),obc):
            new_path.append(path[i])
    #new_path.append(path[-1])
    return new_path

def steerTo(start, end, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # test if there is a collision free path from start to end, with step size
    # given by step_sz, and with generic collision check function
    # here we assume start and end are tensors
    # return 0 if in coliision; 1 otherwise
    start_t = time.time()
    DISCRETIZATION_STEP=step_sz
    #print('start:')
    #print(start)
    #print('end:')
    #print(end)
    #start_t = time.time()
    delta = end - start  # change
    if not isinstance(delta, np.ndarray):
        # then is torch tensor
        delta = delta.numpy()
    # instead of using Euclidean distance, use the l1-norm
    total_dist = np.absolute(delta).max()
    #total_dist = np.linalg.norm(delta)
    # obtain the number of segments (start to end-1)
    # the number of nodes including start and end is actually num_segs+1
    num_segs = int(total_dist / DISCRETIZATION_STEP)
    if num_segs == 0:
        # distance smaller than threshold, just return 1
        return 1
    # obtain the change for each segment
    delta_seg = delta / num_segs
    # initialize segment
    if not isinstance(start, np.ndarray):
        # then is torch tensor
        seg = start.numpy()
    else:
        seg = start
    # check for each segment, if they are in collision
    for i in range(num_segs+1):
        #print(seg)
        if IsInCollision(seg, obc):
            # in collision
            #print(time.time()-start_t)
            return 0
        seg = seg + delta_seg
    #print('steerTo time: ')
    #print(time.time()-start_t)
    return 1

def feasibility_check(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # checks the feasibility of entire path including the path edges
    # by checking for each adjacent vertices
    print('inside feasible check: path length: %d' % (len(path)))
    print(path)
    for i in range(0,len(path)-1):
        if not steerTo(path[i],path[i+1],obc,IsInCollision,step_sz=step_sz):
            # collision occurs from adjacent vertices
            return 0
    return 1

def lvc(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # lazy vertex contraction
    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(path[i],path[j],obc,IsInCollision,step_sz=step_sz)
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])
                return lvc(pc,obc,IsInCollision,step_sz=step_sz)
    return path

def neural_replan(mpNet, path, obc, obs, IsInCollision, normalize, unnormalize, init_plan_flag, step_sz=DEFAULT_STEP, \
                time_flag=False, device=torch.device('cpu')):
    if init_plan_flag:
        # if it is the initial plan, then we just do neural_replan
        MAX_LENGTH = 80
        mini_path, time_d = neural_replanner(mpNet, path[0], path[-1], obc, obs, IsInCollision, \
                                            normalize, unnormalize, MAX_LENGTH, step_sz=step_sz, device=device)
        if mini_path:
            if time_flag:
                return removeCollision(mini_path, obc, IsInCollision), time_d
            else:
                return removeCollision(mini_path, obc, IsInCollision)
        else:
            # can't find a path
            if time_flag:
                return path, time_d
            else:
                return path
    MAX_LENGTH = 50
    # replan segments of paths
    new_path = [path[0]]
    time_norm = 0.
    for i in range(len(path)-1):
        # look at if adjacent nodes can be connected
        # assume start is already in new path
        start = path[i]
        goal = path[i+1]
        steer = steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)
        if steer:
            new_path.append(goal)
        else:
            # plan mini path
            mini_path, time_d = neural_replanner(mpNet, start, goal, obc, obs, IsInCollision, \
                                                normalize, unnormalize, MAX_LENGTH, step_sz=step_sz, device=device)
            #print('mini path length: %d' % (len(mini_path)))
            time_norm += time_d
            if mini_path:
                new_path += removeCollision(mini_path[1:], obc, IsInCollision)  # take out start point
            else:
                new_path += path[i+1:]     # just take in the rest of the path
                break
    if time_flag:
        return new_path, time_norm
    else:
        return new_path


def neural_replanner(mpNet, start, goal, obc, obs, IsInCollision, normalize, unnormalize, MAX_LENGTH, step_sz=DEFAULT_STEP,\
                    device=torch.device('cpu')):
    # plan a mini path from start to goal
    # obs: tensor
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    new_path = []
    time_norm = 0.
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        if tree==0:
            ip1=torch.cat((obs,start,goal)).unsqueeze(0)
            #print('before normalizing:')
            #print(ip1)
            # firstly we need to normalize in order to input to network
            #print('before normalizating...')
            #print(ip1)
            time0 = time.time()
            ip1=normalize(ip1)
            time_norm += time.time() - time0
            #print('after normalizing...')
            #print(ip1)
            #print('after unnormalizationg....')
            #print(unnormalize(torch.tensor(ip1)))
            ip1=to_var(ip1, device)
            start=mpNet(ip1).squeeze(0)
            # unnormalize to world size
            start=start.data.cpu()
            #print('before unnormalizing..')
            #print(start)
            time0 = time.time()
            start = unnormalize(start)
            time_norm += time.time() - time0
            #print('after unnormalizing:')
            #print(start)
            pA.append(start)
            tree=1
        else:
            ip2=torch.cat((obs,goal,start)).unsqueeze(0)
            time0 = time.time()
            ip2=normalize(ip2)
            time_norm += time.time() - time0
            ip2=to_var(ip2, device)
            goal=mpNet(ip2).squeeze(0)
            # unnormalize to world size
            goal=goal.data.cpu()
            time0 = time.time()
            goal = unnormalize(goal)
            time_norm += time.time() - time0
            pB.append(goal)
            tree=0
        target_reached=steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)

    if target_reached==0:
        return 0, time_norm
    else:
        for p1 in range(len(pA)):
            new_path.append(pA[p1])
        for p2 in range(len(pB)-1,-1,-1):
            new_path.append(pB[p2])
    return new_path, time_norm

def transformToTrain(path, path_length, obs, obs_i):
    dataset=[]
    targets=[]
    env_indices = []
    for m in range(0, path_length-1):
        data = np.concatenate( (path[m], path[path_length-1]) ).astype(np.float32)
        targets.append(path[m+1])
        dataset.append(data)
        env_indices.append(obs_i)
    return dataset,targets,env_indices
