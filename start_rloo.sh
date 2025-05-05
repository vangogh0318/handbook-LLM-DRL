set -x

#sleep 3h
#python ./rloo.py --config recipes/Qwen2.5-0.5B-Instruct/rloo/config_demo_accreward_simpledata_newdata.yaml
#python ./filter_data.py --config recipes/Qwen2.5-0.5B-Instruct/rloo/config_demo_accreward_simpledata.yaml
#python ./test.py --config recipes/Qwen2.5-0.5B-Instruct/rloo/config_demo_accreward_simpledata.yaml

python ./rloo2.py --config recipes/Qwen2.5-0.5B-Instruct/rloo/config_demo_accreward_simpledata_copygrpo.yaml
