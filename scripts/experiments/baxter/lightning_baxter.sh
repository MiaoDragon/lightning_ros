python lightning_client.py \
--env_data_path ../../../data/baxter/env/environment_data/ --path_data_path ../../../data/baxter/train/paths/ --pointcloud_data_path ../../../data/baxter/test/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainPaths.pkl \
--good_path_sample_path ../../../results/baxter/path_samples/good_path_samples --bad_path_sample_path ../../../results/baxter/path_samples/bad_path_samples \
--NP 1000 --N 10
