set -x

#sleep 3h
#python -m pdb ./grpo.py --config recipes/Qwen2.5-0.5B-Instruct/grpo/config_demo.yaml
#python ./grpo.py --config recipes/Qwen2.5-0.5B-Instruct/grpo/config_demo_accreward.yaml
#python ./grpo.py --config recipes/Qwen2.5-0.5B/grpo/config_demo_accreward.yaml
#python ./grpo.py --config recipes/Qwen2.5-0.5B-Instruct/grpo/config_demo_numgen8.yaml
python ./grpo.py --config recipes/Qwen2.5-0.5B-Instruct/grpo/config_demo_accreward_simpledata.yaml
#python ./grpo.py --config recipes/Qwen2.5-0.5B/grpo/config_demo_accreward_simpledata.yaml
