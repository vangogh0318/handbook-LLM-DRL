
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=model/Qwen-2.5-0.5B-Simple-RL
MODEL=model/Qwen2.5-0.5B
MODEL=model/Qwen2.5-0.5B-Instruct
MODEL=model/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=model/Qwen2.5-0.5B-RL-num_gen4/checkpoint-117
MODEL=model/Qwen2.5-0.5B-RL-num_gen4_accreward/checkpoint-117
MODEL=model/Qwen2.5-0.5B-base-RL-num_gen4_accreward/checkpoint-117
MODEL=model/Qwen2.5-0.5B-RL-numgen8_accreward/checkpoint-234/
MODEL=model/Qwen2.5-0.5B-RL-numgen8_accreward/checkpoint-120/
MODEL=model/Qwen2.5-0.5B-RL-numgen8_accreward/checkpoint-20/
MODEL=model/Qwen-2.5-0.5B-SimpleRL-Zoo
MODEL=model/Qwen2.5-0.5B-simpleRL-reproduce/checkpoint-127/
MODEL=model/Qwen2.5-0.5B-RL-num_gen4_accreward/checkpoint-117
MODEL=model/Qwen2.5-0.5B-simpleRL-reproduce-lr5e7/checkpoint-127
MODEL=model/Qwen2.5-0.5B-simpleRL-reproduce-lr5e7-epoch2
MODEL=model/Qwen2.5-0.5B-simpleRL-rloo/checkpoint-1018/

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
