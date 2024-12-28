import json
import random
import time
import requests
import multiprocessing
from tqdm import tqdm
import re
import argparse
import os

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["mistral","qwen-2.5-7b-instruct","llama3.1-8b-instruct","llama2-7b-chat-4k", "llama-2-7B-32k-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--hopf_type', type=str, default="max_fused")
    parser.add_argument('--len', "-l", type=int, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--window_size", "-ws", type=int, default=3, help="Window size for HopFormer")
    parser.add_argument("--sim_threshold", "-st", type=float, default=20.0, help="Similarity threshold for HopFormer")
    parser.add_argument("--num_attn_sinks", "-snks", type=float, default=0, help="Attention sinks (streaming LLM)")
    parser.add_argument("--gumbel", "-gbl", action='store_true', help="use gumbel softmax")
    parser.add_argument("--no_hopf", action='store_true', help="Disable HopFormer")  # Updated line
    parser.add_argument("--save_wts", action='store_true', help="Save attn wts")  # Updated line
    return parser.parse_args(args)

dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]
# model = "LongWriter-glm4-9b"
args = parse_args()
model_name = args.model

prompt_template = open("judge.txt", "r", encoding="utf-8").read()

GPT4_API_KEY = 'Mue2YhdKumdycUEzCQjeiVPzOJD0FEPN' # Your API Key
GPT_MODEL = 'mistral-large-latest'
def get_response_gpt4(prompt, temperature=0.5, max_new_tokens=1024, stop=None):
    tries = 0
    while tries < 10:
        # import pdb; pdb.set_trace()
        tries += 1
        try:
            # headers = {
            #     'Authorization': "Bearer {}".format(GPT4_API_KEY),
            # }
            headers = {
                "Authorization": f"Bearer {GPT4_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            messages = [
                {'role': 'user', 'content': prompt},
            ]
            #https://api.openai.com/v1/chat/completions
            resp = requests.post("https://api.mistral.ai/v1/chat/completions", json = { 
                "model": GPT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                # "stop": stop,
            }, headers=headers, timeout=600)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            # time.sleep(60)
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy'
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
    else:
        print("Max tries. Failed.")
        return "Max tries. Failed."
    try:
        return resp["choices"][0]["message"]["content"]
    except: 
        return ''

def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def process_data(items, fout):
    for item in tqdm(items):
        prompt = prompt_template.replace('$INST$', item['prompt']).replace('$RESPONSE$', item["response"])
        scores = None
        trys = 0
        while scores is None and trys < 5:
            output = get_response_gpt4(prompt)
            try:
                if '```json' in output:
                    output = extract_info(r'```json\n(.*?)\n```', output)
                output = output.replace('\n', '')
                scores = json.loads(output)
                for dim in dims:
                    if dim not in scores:
                        scores = None
                        trys += 1
            except Exception as e:
                trys += 1
        if scores is None:
            print(output)
        else:
            item['scores'] = scores
            fout.write(json.dumps(item, ensure_ascii=False)+'\n')
            fout.flush()

res_path = f"preds/{model_name}/"
result = dict()
for file in os.listdir(res_path):
    if "judge" in file or "log" in file or "png" in file or "result" in file:
        continue
    try:
        data = [json.loads(line) for line in open(f"{res_path}{file}", encoding='utf-8')]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file}: {e}")
        continue
    random.shuffle(data)
    # PROC_NUM = 8
    # pool = multiprocessing.Pool(processes=PROC_NUM)
    # total = len(data)

    # for i in range(PROC_NUM):
    #     start = (i * total) // PROC_NUM
    #     end = None if i == PROC_NUM - 1 else ((i + 1) * total) // PROC_NUM
    # import pdb; pdb.set_trace()
    if f"{file[:-6]}_judge.jsonl" not in os.listdir(res_path):
        print("Using LLM Judge!\n")
        fout = open(f"{res_path}{file[:-6]}_judge.jsonl", 'w', encoding='utf-8')
        process_data(data, fout)
    # pool.close()
    # pool.join()
    # fout.close()

    all_scores = [json.loads(line)['scores'] for line in open(f"{res_path}{file[:-6]}_judge.jsonl", 'r', encoding='utf-8')]

    total_score = dict()
    for dim in dims:
        scores = [float(score[dim]) if dim in score else 3 for score in all_scores]
        total_score[dim] = ((sum(scores) / len(scores)) - 1) * 25
    total_score['total'] = sum(total_score.values()) / len(total_score)
    result[file[:-6]] = total_score

json.dump(result, open(f"preds/{model_name}/judge_result.json","w"), ensure_ascii=False, indent=4)
