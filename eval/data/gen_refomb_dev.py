
import json
import os
import time
from openai import OpenAI

# 初始化 OpenAI 客户端 (需要提前设置环境变量 OPENAI_API_KEY)
# client = OpenAI()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def process_jsonl_with_gpt4o(input_file, output_file):
    """
    处理 JSONL 文件，调用 GPT-4o 生成答案并保存结果
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    
    # 创建备份文件 (安全考虑)
    timestamp = int(time.time())
    backup_file = f"{os.path.splitext(input_file)[0]}_backup_{timestamp}.jsonl"
    with open(backup_file, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 处理每条数据
    for i, item in enumerate(items):
        print(f"Processing item {i+1}/{len(items)}...")
        
        # 准备 API 调用参数
        prompt = (
            "You are an expert visual assistant. "
            "Answer the user's question about the image in a concise and accurate manner. "
            "Describe ONLY what you can see in the image. "
            "Do NOT speculate about context not visible. "
            "Provide short and factual responses. "
            f"QUESTION: {item['question']}\n"
            f"IMAGE URL: {item['image_url']}"
        )
        
        # 调用 GPT-4o
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a factual visual assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            # 提取并存储答案
            answer = response.choices[0].message.content.strip()
            item["answer"] = answer
            
            # 每隔5条保存一次进度 (防止中断丢失数据)
            if (i+1) % 5 == 0:
                save_temp_result(items, output_file)
                
            time.sleep(1.2)  # 避免API速率限制
                
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            item["answer"] = "ERROR: " + str(e)
    
    # 保存最终结果
    save_temp_result(items, output_file)
    print(f"Processing completed! Results saved to {output_file}")

def save_temp_result(items, output_file):
    """临时保存处理结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_path = "eval/data/RefoMB_dev.jsonl"  # 替换为实际文件路径
    output_path = input_path  # 直接覆盖原文件
    
    # 调用处理函数
    process_jsonl_with_gpt4o(input_path, output_path)