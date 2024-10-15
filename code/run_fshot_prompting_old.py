import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import argparse

# 1. Define the Custom Dataset
class LabelDatasetSingleChoice(Dataset):
    def __init__(self, data, labels, prompt_func):
        """
        Args:
            data (list): List of examples.
            labels (list): List of possible labels.
            prompt_func (callable): Function to construct prompts.
        """
        self.data = data
        self.labels = labels
        self.prompt_func = prompt_func
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = self.prompt_func(example, self.labels)
        return {
            "example_id": example.get("id", idx),  # Ensure each example has a unique ID
            "prompt": prompt,
            "true_label": example["label"]
        }

def construct_prompt_single_choice(example, labels):
    # Create a comma-separated list of labels
    # print('example:', example)

    label_list = ", ".join(labels)

    label_text = ""

    for label_idx, label in enumerate(labels):
        label_text += f"{label_idx}: {label}\n"


    prompt = f'''Given the sentence: "{example['orig_sent']}", which one of the following relations between the two entities <e1> and <e2> is being discussed?\n Choose one from this list of {len(labels)} options:\n{label_text}\nThe answer is : '''


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
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False
        )
    
    # Decode the generated sequences
    generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    return generated_texts



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang",       help="choice of source language",           type=str, default='en')
    parser.add_argument('--dataset', 	    help='choice of dataset', 			        type=str, default='indore')

    parser.add_argument("--batch_size", 										        type=int, default=16)
    parser.add_argument('--model_name', 	help='name of the LL to use', 	            type=str, default='llama3')
    parser.add_argument('--dep_parser', 	help='name of the dependecy parser', 	    type=str, default='stanza')
    parser.add_argument('--split', 	        help='split', 	                            type=str, default='test')
    parser.add_argument('--mode',           help='mode', 	                            type=str, default='zshot')

    # default parameters
    
    args                  = parser.parse_args()
    
    return args

if __name__ =='__main__':	
    args                            =   get_args()

    annot_file                      =   f'../data/{args.dataset}/{args.src_lang}_prompt_{args.dep_parser}.json'

    with open(annot_file) as f:
        data = json.load(f)

    split_data = data.get(args.split, [])

    # List of possible labels
    with open(f"../data/{args.dataset}/relation_dict.json") as f:
        relation_data = json.load(f)

    # labels_with_underscore = list(relation_data.values())
    # labels = [s.replace("_", " ") for s in labels_with_underscore]
    labels = list(relation_data.values())

    # Create the dataset
    dataset = LabelDatasetSingleChoice(
        data=split_data,
        labels=labels,
        prompt_func=construct_prompt_single_choice
    )

    # dataset     = dataset[:10]

    # Define DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_name == 'llama3':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model_name == 'gemma2':
        model_id = 'google/gemma-2-9b-it'
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Assign pad_token to eos_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("pad_token was not set. Using eos_token as pad_token.")

    # Load model with 8-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()  # Set model to evaluation moder
    results = []
    count = 0
    for batch in tqdm(dataloader, desc="Generating Responses"):
        count       += 1
        prompts     = batch["prompt"]
        example_ids = batch["example_id"]
        true_labels = batch["true_label"]

        # Generate responses
        generated_texts = generate_responses(prompts, model, tokenizer)


        # Map responses
        for ex_id, gen_text in enumerate(generated_texts):
            # Clean and normalize the generated text
            curr_data = {}
            curr_data['id']              = count
            curr_data['true_label']      = true_labels[ex_id]
            curr_data['gen_text']        = gen_text.replace(prompts[ex_id], '')
            curr_data['prompt']          = prompts[ex_id]
            count += 1
            results.append(curr_data)

            # print(f"Prompt : {prompts[ex_id]}")
            # print(f"True Label : {true_labels[ex_id]}")
            # print(f"Generated Text : {gen_text.replace(prompts[ex_id], '')}")
            # print()
            
        
    print("Generation completed.")

    with open(f'../prompting_predictions/{args.dataset}-{args.src_lang}-{args.model_name}_zshot_simple_prompt-{args.split}.json', 'w') as f:
        json.dump(results, f, indent=4)