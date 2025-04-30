#model
#huggingface-cli download --resume-download --local-dir-use-symlinks False cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr --local-dir cleanrl_EleutherAI_pythia-1b-deduped__sft__tldr
#huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen2.5-0.5B-Instruct
#huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen2.5-1.5B-Instruct
#huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-0.5B --local-dir model/Qwen2.5-0.5B
#huggingface-cli download --resume-download --local-dir-use-symlinks False deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir model/DeepSeek-R1-Distill-Qwen-1.5B
#huggingface-cli download --resume-download --local-dir-use-symlinks False hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo --local-dir model/Qwen-2.5-0.5B-SimpleRL-Zoo

#datasets
#huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
#huggingface-cli download --repo-type dataset --token dddtoken --resume-download trl-internal-testing/descriptiveness-sentiment-trl-style
#huggingface-cli download --repo-type dataset --token dddtoken --resume-download trl-internal-testing/descriptiveness-sentiment-trl-style
#huggingface-cli download --repo-type dataset --token dddtoken --resume-download Dahoas/rm-static
#huggingface-cli download --repo-type dataset --resume-download DigitalLearningGmbH/MATH-lighteval --local-dir datasets/MATH-lighteval

huggingface-cli download --repo-type dataset --resume-download hkust-nlp/SimpleRL-Zoo-Data --local-dir datasets/SimpleRL-Zoo-Data
