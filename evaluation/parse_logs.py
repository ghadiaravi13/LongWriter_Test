import argparse
import datetime
import json
import os
import pandas as pd

from networkx import cut_size

attn_heads_dict = {"llama3.1-8b-instruct": 32, "phi4": 40, "mistral": 32, "qwen2.5": 28, "phi4-unsloth": 40}
layers_dict = {"llama3.1-8b-instruct": 32, "phi4": 40, "mistral": 32, "qwen2.5": 28, "phi4-unsloth": 40}
key_heads_dict = {"llama3.1-8b-instruct": 8, "phi4": 10, "mistral": 8, "qwen2.5": 4, "phi4-unsloth": 10}
hid_dim_dict = {"llama3.1-8b-instruct": 128, "phi4": 128, "mistral": 128, "qwen2.5": 128, "phi4-unsloth": 128}



def parse(args,dataset=""):
    model = args.model
    files = sorted(os.listdir(f"preds/{model}/"),key=lambda x: "hopf_False" not in x)
    if args.e:
        files = sorted(os.listdir(f"preds_e/{model}/"),key=lambda x: "hopf_False" not in x)

    for file in files:
        if ".log" in file and dataset in file:
            if args.e:
                logfile = open(f"preds_e/{model}/"+file,"r")
                res = open(f"preds_e/{model}/"+file[:-3]+"jsonl","r").readlines()
            else:
                logfile = open(f"preds/{model}/"+file,"r")
                res = open(f"preds/{model}/"+file[:-3]+"jsonl","r").readlines()
            last_time = 0
            run_name = file.split("/")[-1][:-4]
            
            n_attn_heads = key_heads_dict[model]
            n_key_heads = key_heads_dict[model] if "no_hopf" in file else n_attn_heads
            n_layers = layers_dict[model]
            hid_dim_per_head = hid_dim_dict[model]
            
            if "hopf_False" in file:
                data = dict()
            data[run_name] = dict()
            data[run_name]['gen_time(s)'] = []
            data[run_name]['inp_size'] = [] if "hopf_False" in file else inp_size
            data[run_name]['out_size'] = [] if "hopf_False" in file else out_size
            data[run_name]['resp_len'] = []
            data[run_name]['KV_size'] = []
            data[run_name]['attn_size'] = []
            data[run_name]['total_cache'] = []

            i = 0
            for l in logfile.readlines():
                if(l!="\n"):
                    curr_time = datetime.datetime.strptime(l.split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f").timestamp()
                    data[run_name]['gen_time(s)'].append(curr_time - last_time)
                    if "hopf_False" in file:
                        data[run_name]['inp_size'].append(int(l.split("Input size: ")[1].split(" ")[0]))
                        data[run_name]['out_size'].append(int(l.split("key': ")[1].split(",")[0]) - int(l.split("Input size: ")[1].split(" ")[0]))
                    data[run_name]['resp_len'].append(int(l.split("len': ")[1].split("}")[0]))
                    data[run_name]['KV_size'].append(4 * hid_dim_per_head * n_key_heads * n_layers * int(l.split("key': ")[1].split(",")[0]) / 1000000)
                    # if "hopf_False" in file:
                    #     data[run_name]['attn_size'].append(4 * n_attn_heads * n_layers * data[run_name]['out_size'][i] * int(l.split("attn_wts': ")[1].split("}")[0]) / 1000000)
                    # else:
                    data[run_name]['attn_size'].append(4 * n_attn_heads * n_layers * data[run_name]['resp_len'][i] * int(l.split("attn_wts': ")[1].split(",")[0]) / 1000000)
                    data[run_name]['total_cache'].append(data[run_name]['KV_size'][-1] + data[run_name]['attn_size'][-1])
                    last_time = curr_time
                    i+=1
            inp_size = data[run_name]['inp_size']
            out_size = data[run_name]['out_size']
    return data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert results JSON to CSV format')
    parser.add_argument("--model",default=None, type=str, help='Model name for the result parsing')
    parser.add_argument("--e",action='store_true',help='results for longbench-E')

    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert file
    # try:
    data = parse(args)
    df = pd.concat({k: pd.DataFrame(v) for k, v in data.items()}, axis=1)

    if args.e:
        with open(f"preds_e/{args.model}/parsed_log_data.json","w") as f:
            json.dump(data,f)
            try:
                with pd.ExcelWriter(f"preds_e/parsed_log_data.xlsx",mode='a') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"preds_e/parsed_log_data.xlsx",mode='w') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            print(f"Successfully parsed logs")
        f.close()
    else:
        with open(f"preds/{args.model}/parsed_log_data.json","w") as f:
            json.dump(data,f)
            try:
                with pd.ExcelWriter(f"preds/parsed_log_data.xlsx",mode='a') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"preds/parsed_log_data.xlsx",mode='w') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            print(f"Successfully parsed logs")
        f.close()
    # except Exception as e:
    #     print(f"Error converting file: {str(e)}")

if __name__ == "__main__":
    main()
    



                