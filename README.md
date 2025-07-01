## Evaluation steps
### 1. Activate checkpoints server
'''
VLLM_LOGGING_LEVEL=DEBUG vllm serve emmazhao118/llava-finetuned-rlaifv --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=10,video=5
'''

### 2. Get response from the server using test dataset
'''
pip install openai==1.68.2
python eval/data/gen_refomb_dev.py
'''

### 3. Get overall score based on RefoMB
'''
pip install openai==0.28
save_dir="eval/results"
model_ans_path="eval/data/RefoMB_dev.jsonl"
model_name=<model_name> # e.g. "output/llava1_7b_lora_orpo/v5"
bash script/eval/run_refomb_overall.sh $save_dir $model_ans_path $model_name
'''

### 4. Get hallucination score based on RefoMB
'''
eval_result="eval/results/A-GPT-4V_B-output/llava1_7b_lora_orpo/v5"
bash script/eval/run_refomb_hall.sh $eval_result
'''