XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    -v $PWD:/root/catkin_ws/src/lightning_ros/ \
    --mount type=bind,source=/media/arclabdl1/HD1/YLmiao/data,target=/root/catkin_ws/src/lightning_ros/data,readonly \
    --mount type=bind,source=/media/arclabdl1/HD1/YLmiao/results/lightning_res,target=/root/catkin_ws/src/lightning_ros/results \
    --runtime=nvidia \
    lightning_mpnet \
    bash

# add this under -v when using GPU    --runtime=nvidia \
