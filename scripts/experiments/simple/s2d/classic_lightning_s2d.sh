# this is for testing and compare against previous simple experiments
#cd ..
#python lightning_client.py --res_path ../../../results/simple/s2d/seen/ --N 100 --NP 200 --data_path ../../../data/simple/s2d/ \
#--env_type s2d --env_idx 0 --path_idx 4000
# 100 4000
#cd s2d
cd ..
python lightning_client.py --res_path ../../../results/simple/s2d/unseen/ --N 10 --NP 2000 --data_path ../../../data/simple/s2d/ \
--env_type s2d --env_idx 100 --path_idx 0
# 100 4000
cd s2d
