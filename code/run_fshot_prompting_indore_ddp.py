import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import argparse
import re
import networkx as nx
from collections import deque
import advertools as adv
import os
from natsort import natsorted

with open('dependency_mapping.json', 'r') as file:
    dependency_definitions = json.load(file)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 1. Define the Custom Dataset
class LabelDatasetSingleChoice(Dataset):
    def __init__(self, data, labels, prompt_func, stop_words, dependency_definitions):
        """
        Args:
            data (list): List of examples.
            labels (list): List of possible labels.
            prompt_func (callable): Function to construct prompts.
        """
        self.data = data
        self.labels = labels
        self.prompt_func = prompt_func
        self.stop_words = stop_words
        self.dependency_definitions = dependency_definitions
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = self.prompt_func(example, self.labels, self.stop_words, self.dependency_definitions)
        return {
            "example_id": example.get("id", idx),  # Ensure each example has a unique ID
            "prompt": prompt,
            "true_label": example["label"]
        }

# BFS function to find the first entry point for an entity starting from the root
def find_first_entry_point(graph, root, entity_words):
    queue = deque([root])
    visited = set()
    
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        
        # Check if the node is part of the entity words
        if node in entity_words:
            return node
        
        # Add neighbors to the queue for BFS
        queue.extend(graph.predecessors(node))
    
    return None  # If no entry point is found

def find_entity_connecting_path(e1,e2,dep_graph):
    G = nx.DiGraph()

    # Add nodes and edges with labels
    for relation in dep_graph:
        target, label, source = relation
        G.add_edge(target, source, label=label)

    # Find first entry points for e1 and e2 from the root
    root = "ROOT"
    first_entry_e1 = find_first_entry_point(G, root, e1)
    first_entry_e2 = find_first_entry_point(G, root, e2)

    highlighted_path=[]
    highlighted_path2=[]

    try:
        # Try finding a direct path between first_entry_e1 and first_entry_e2
        path_e1_to_e2 = nx.shortest_path(G, source=first_entry_e1, target=first_entry_e2)
        highlighted_path = path_e1_to_e2  # Direct path found
    except nx.NetworkXNoPath:
        try:
            path_e1_to_e2 = nx.shortest_path(G, source=first_entry_e2, target=first_entry_e1)
            highlighted_path = path_e1_to_e2  # Direct path found
        except nx.NetworkXNoPath:
            try:
                path_root_to_e1 = nx.shortest_path(G, source=first_entry_e1, target=root)
                try:
                    path_e1_to_root = nx.shortest_path(G, source=root, target=first_entry_e1)
                    if len(path_root_to_e1<path_e1_to_root):
                        highlighted_path=path_root_to_e1
                    else:
                        highlighted_path=path_e1_to_root
                except:
                    highlighted_path=path_root_to_e1
            except:
                pass

            try:
                path_root_to_e2 = nx.shortest_path(G, source=first_entry_e2, target=root)
                try:
                    path_e2_to_root = nx.shortest_path(G, source=root, target=first_entry_e2)
                    if len(path_root_to_e2<path_e2_to_root):
                        highlighted_path2=path_root_to_e2
                    else:
                        highlighted_path2=path_e2_to_root
                except:
                    highlighted_path2=path_root_to_e2
            except:
                pass

    pruned_parses=[]

    if isinstance(highlighted_path,list):
        for i in range(len(highlighted_path)-1):
            beg=highlighted_path[i]
            end=highlighted_path[i+1]
            if G.has_edge(beg,end):
                rel=G.edges[beg,end].get("label")
            elif G.has_edge(end,beg):
                rel=G.edges[end,beg].get("label")
            pruned_parses.append([beg,rel,end])

    if isinstance(highlighted_path2,list):
        for i in range(len(highlighted_path2)-1):
            beg=highlighted_path2[i]
            end=highlighted_path2[i+1]
            if G.has_edge(beg,end):
                rel=G.edges[beg,end].get("label")
            elif G.has_edge(end,beg):
                rel=G.edges[end,beg].get("label")
            pruned_parses.append([beg,rel,end])

    unique_parses = []
    seen = set()

    for sublist in pruned_parses:
        # Convert the sublist to a tuple so it can be added to a set
        tuple_sublist = tuple(sublist)
        if tuple_sublist not in seen:
            seen.add(tuple_sublist)
            unique_parses.append(sublist)

    return unique_parses


def construct_prompt_single_choice(example, labels,stopwords,dep_choice):
    # Create a comma-separated list of labels
    # print('example:', example)
    label_text = ""
    for key, value in labels.items():
        label_text += f"{key}: {value}\n"


    prompt = f'''Given the sentence: "{example['orig_sent']}", which one of the following relations between the two entities <e1> and <e2> is being discussed?\n Choose one from this list of {len(labels)} options:\n{label_text}\nThe answer is : '''


    return prompt

def construct_prompt_dependency_choice(example, labels):
    # Create a comma-separated list of labels
    # print('example:', example)
    label_text = ""
    for key, value in labels.items():
        label_text += f"{key}: {value}\n"

    dependency_list = example["dep_graph"]
    dep_graph = []
    # (node_dict[n1], rel, node_dict[n2])
    for dep in dependency_list:
        dep_graph.append({'head': dep[2], 'rel': dep[1], 'word': dep[0]})
    
    # import pdb; pdb.set_trace()

    dep_text = json.dumps(dep_graph)
    
    prompt = f'''Given the sentence: "{example['orig_sent']}", which one of the following relations between the two entities <e1> and <e2> is being discussed?\n We also provide the dependency parse in the form of head, rel, and word: {dep_text}\n. Choose one from this list of {len(labels)} options:\n{label_text}\nThe answer is : '''


    return prompt

def construct_prompt_dependency_choice_trimmed(example, labels, stop_words, dependency_definitions):
    # Create a comma-separated list of labels
    # print('example:', example)
    label_text = ""
    for key, value in labels.items():
        label_text += f"{key}: {value}\n"

    dependency_list = example["dep_graph"]
    dep_text = ""
    # (node_dict[n1], rel, node_dict[n2])

    try:

        e1 = re.search(r'<e1>(.*?)</e1>', example['orig_sent']).group(1)
        e2 = re.search(r'<e2>(.*?)</e2>', example['orig_sent']).group(1)
        words_e1 = re.findall(r'\b\w+\b', e1)
        words_e2 = re.findall(r'\b\w+\b', e2)

        #Filter out stop words
        words_e1 = [word for word in words_e1 if word.lower() not in stop_words]
        words_e2 = [word for word in words_e2 if word.lower() not in stop_words]

        pruned_dep_list=find_entity_connecting_path(words_e1,words_e2,dependency_list)

        for dep in pruned_dep_list:
            if dep[1] == "root":
                descriptive_relations = f"{dep[0]} is the root word, "
            else:
                descriptive_relations = f"{dep[0]} is {dependency_definitions.get(dep[1], dep[1])} of {dep[2]}, " 
            dep_text+=descriptive_relations
    except Exception as e:
        dep_text=""
    except KeyboardInterrupt:
        raise
    
    # import pdb; pdb.set_trace()
    
    prompt = f'''Given the sentence: "{example['orig_sent']}", which one of the following relations between the two entities <e1> and <e2> is being discussed?\n We also provide the dependency parses as follows: "{dep_text}"\n Choose one from this list of {len(labels)} options:\n{label_text}\nThe answer is : '''
    #print(prompt)
    return prompt

def generate_responses(prompts, model, tokenizer, max_new_tokens=100):
    # Tokenize the prompts with padding and no truncation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,  # Disable truncation
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.module.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False
        )
    
    # Decode the generated sequences
    generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    return generated_texts

def tensor_to_native(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_native(v) for v in obj]
    return obj



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang",       help="choice of source language",           type=str, default='en')
    parser.add_argument('--dataset', 	    help='choice of dataset', 			        type=str, default='indore')

    parser.add_argument("--batch_size", 										        type=int, default=4)
    parser.add_argument('--model_name', 	help='name of the LL to use', 	            type=str, default='llama3')
    parser.add_argument('--dep_parser', 	help='name of the dependecy parser', 	    type=str, default='stanza') # None if no dependency parser is used
    parser.add_argument('--split', 	        help='split', 	                            type=str, default='test')
    parser.add_argument('--mode',           help='mode', 	                            type=str, default='zshot')
    
    # default parameters
    
    args                  = parser.parse_args()
    
    return args

def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    args                            =   get_args()

    if args.dep_parser != 'None':
        annot_file                      =   f'../data/{args.dataset}/{args.src_lang}_prompt_{args.dep_parser}.json'
    else:
        annot_file                      =   f'../data/{args.dataset}/{args.src_lang}_prompt_trankit.json'

    with open(annot_file) as f:
        data = json.load(f)

    split_data = data.get(args.split, [])

    with open('dependency_mapping.json', 'r') as file:
        dependency_definitions = json.load(file)

    language_dict={
        "en": "english",
        "hi": "hindi",
        "te": "telugu",
        "ar": "arabic",
        "de": "german",
        "es": "spanish",
        "fr": "french",
        "it": "italian",
        "zh": "chinese"
    }

    language=language_dict[args.src_lang]

    stop_words = adv.stopwords[language]

    # List of possible labels
    with open(f"../data/{args.dataset}/relation_dict.json") as f:
        relation_labels = json.load(f)

    labels= dict(natsorted(relation_labels.items()))

    dataset = LabelDatasetSingleChoice(
        data=split_data,
        labels=labels,
        prompt_func=construct_prompt_dependency_choice_trimmed if args.dep_parser.lower() != 'none' else construct_prompt_single_choice,
        stop_words=stop_words,
        dependency_definitions=dependency_definitions
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )
        # Define DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,shuffle=False)

    # Map model names to Hugging Face model IDs
    model_id_mapping = {
        'llama3': "meta-llama/Meta-Llama-3-8B-Instruct",
        'gemma2': 'google/gemma-2-9b-it',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
        'qwen':'Qwen/Qwen2-7B-Instruct'
        }

    model_id = model_id_mapping.get(args.model_name.lower())
    if model_id is None:
        raise ValueError(f"Unsupported model_name '{args.model_name}'. Choose from {list(model_id_mapping.keys())}.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": rank}
    )
    model = DDP(model, device_ids=[rank])
    model.eval()

    results = []
    count = 0
    for batch in tqdm(dataloader, desc=f"Generating Responses (GPU {rank})"):
        try:
            prompts = batch["prompt"]
            example_ids = batch["example_id"]
            true_labels = batch["true_label"]

            generated_texts = generate_responses(prompts, model, tokenizer)

            for ex_id, gen_text, prompt, true_label in zip(example_ids, generated_texts, prompts, true_labels):
                cleaned_gen_text = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text.strip()
                results.append({
                    'id': ex_id,
                    'true_label': true_label,
                    'gen_text': cleaned_gen_text,
                    'prompt': prompt
                })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing batch on GPU {rank}: {e}")
            continue

    # Gather results from all processes
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, results)

    if rank == 0:
        # Combine and save results
        combined_results = [item for sublist in all_results for item in sublist]
        combined_results = tensor_to_native(combined_results)
        output_filename = f'../prompting_predictions/{args.dataset}-{args.src_lang}-{args.model_name}_zshot_prompt-{args.dep_parser}-{args.split}-better_prompt.json'
        with open(output_filename, 'w') as f:
            json.dump(combined_results, f, indent=4)
        print(f"Results saved to {output_filename}")
    cleanup()

if __name__ == '__main__':
    args = get_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
