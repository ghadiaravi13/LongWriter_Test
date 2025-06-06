import logging
from pyexpat import model
from regex import cache_all
import requests
import time, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np
import random
import codecs
import argparse
from copy import deepcopy
from tqdm import tqdm
import traceback
import re
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['HF_HOME'] = "/work/10198/ghadiaravi13/ls6/HopFormer/HF_Llama3/HF_cache"
cache_dir = "/work/10198/ghadiaravi13/ls6/HopFormer/HF_Llama3/HF_cache/"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["phi4-unsloth","phi4","mistral","qwen2.5","llama3.1-8b-instruct","llama2-7b-chat-4k", "llama-2-7B-32k-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--hopf_type', type=str, default="max_fused")
    parser.add_argument('--len', "-l", type=int, default=None)
    parser.add_argument("--window_size", "-ws", type=int, default=3, help="Window size for HopFormer")
    parser.add_argument("--exhale_after", "-ea", type=float, default=1.0, help="Exhale after exceeding this times the KV limit")
    parser.add_argument("--sim_threshold", "-st", type=float, default=20.0, help="Similarity threshold for HopFormer")
    parser.add_argument("--num_attn_sinks", "-snks", type=float, default=0, help="Attention sinks (streaming LLM)")
    parser.add_argument("--gumbel", "-gbl", action='store_true', help="use gumbel softmax")
    parser.add_argument("--no_hopf", action='store_true', help="Disable HopFormer")  # Updated line
    parser.add_argument("--save_wts", action='store_true', help="Save attn wts")  # Updated line
    return parser.parse_args(args)

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def get_pred(data, path, max_new_tokens, temperature, tokenizer, fout, args):
    device = torch.device(f'cuda')
    logger = logging.getLogger(__name__)
    # model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # Create a custom configuration with HopFormer parameters
    config = AutoConfig.from_pretrained(path, cache_dir=cache_dir)
    config._attn_implementation = "flash_attention_2"
    if args.hopf_type=='indp':
        assert args.window_size==1, "Window size > 1 is not supported for independent Hopformer. Try 'max_fused' instead"
    config.hopformer = None if args.no_hopf or args.hopf_type=="snapkv" else {
        'window_size': int(args.window_size),
        'sim_threshold': int(args.sim_threshold),
        'softmax': 'gumbel' if args.gumbel else 'normal',
        'num_attn_sinks': int(args.num_attn_sinks),
        'hopf_type': args.hopf_type,
        'exhale_after': args.exhale_after
    }
    print(f"Hopformer is: {config.hopformer}")
    if args.hopf_type=="snapkv":
        config.snapkv = True
        config.window_size = args.window_size
        config.max_capacity_prompt = args.sim_threshold
        config.kernel_size = 5
        config.pooling = "avgpool"
    else:
        config.snapkv = False

    # Load the model with the custom configuration
    model = AutoModelForCausalLM.from_pretrained(path,
                                                config=config,
                                                torch_dtype=torch.bfloat16,
                                                cache_dir=cache_dir).to(device)
    
    model = model.eval()
    i = 0
    with torch.no_grad():
        for dt in tqdm(data):
            if i>8 or dt['length']>3000:
                prompt = dt['prompt']
                if "llama" in path.lower() or "mistral" in path.lower() or "phi" in path.lower() or "qwen" in path.lower():
                    prompt = f"[INST]{prompt}[/INST]"
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                    context_length = input.input_ids.shape[-1]
                    if "phi" in path.lower():
                        output = model.generate(
                            **input,
                            max_new_tokens=min(max_new_tokens,dt['length']*2),
                            num_beams=1,
                            do_sample=True,
                            temperature=temperature,
                            min_length=context_length+1,
                        )[0]
                    else:
                        output = model.generate(
                            **input,
                            max_new_tokens=min(max_new_tokens,dt['length']*2),
                            num_beams=1,
                            do_sample=True,
                            temperature=temperature
                        )[0]
                    os.makedirs("/work/10198/ghadiaravi13/vista/HopFormer/LongWriter_Test/evaluation/attn_wts/", exist_ok=True)    
                    torch.save(tokenizer.convert_ids_to_tokens(output),f"/work/10198/ghadiaravi13/vista/HopFormer/LongWriter_Test/evaluation/attn_wts/qwen_ws{args.window_size}_st{args.sim_threshold}_fused{args.hopf_type}_tokenized_seq.pt")
                    response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                else:
                    response, history = model.chat(tokenizer, prompt, history=[], max_new_tokens=max_new_tokens, temperature=temperature)
                dt["response_length"] = count_words(response)
                dt["response"] = response
                fout.write(json.dumps(dt, ensure_ascii=False)+'\n')
                fout.flush()
                break
            i+=1
            # print(response)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open('/work/10198/ghadiaravi13/vista/HopFormer/LongBench_Test/config/model2path.json','r'))
    model_name = args.model#'LongWriter-glm4-9b' # LongWriter-llama3.1-8b
    path = model2path[model_name]#"THUDM/LongWriter-glm4-9b" # THUDM/LongWriter-llama3.1-8b
    os.makedirs(f"preds/{model_name}", exist_ok=True)
    fout = open(f"preds/{model_name}/profile_preds_ws{args.window_size}_st{args.sim_threshold}_ea{args.exhale_after}_snks{args.num_attn_sinks}_hopf_{not(args.no_hopf)}_type_{args.hopf_type}_len{args.len}_gbl{args.gumbel}.jsonl", 'w', encoding='utf-8')
    
    logfile = f"preds/{model_name}/profile_preds_ws{args.window_size}_st{args.sim_threshold}_ea{args.exhale_after}_snks{args.num_attn_sinks}_hopf_{not(args.no_hopf)}_type_{args.hopf_type}_len{args.len}_gbl{args.gumbel}.log"
    logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
    
    max_new_tokens = 32768
    temperature = 0.5
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)

    with open('longbench_write_en.jsonl', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    get_pred(data, path, max_new_tokens, temperature, tokenizer, fout, args)