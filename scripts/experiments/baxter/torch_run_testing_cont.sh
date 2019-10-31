python torch_test_cont.py \
--env_data_path ./env/environment_data/ --path_data_path ../../../data/baxter/test/paths/ --pointcloud_data_path ../../../data/baxter/test/pcd/ \
--envs_file trainEnvironments.pkl --path_data_file trainEnvironments_testPaths.pkl \
--model_path ../../../results/baxter/pfs_rr/ --model_name model_baxter.pkl --experiment_name lightning_pfs_rr --device 2 --N 10 --NP 100 \
--good_path_sample_path ../../../results/baxter/test/pfs_rr/path_samples/good_path_samples --bad_path_sample_path ../../../results/baxter/test/pfs_rr/path_samples/bad_path_samples
