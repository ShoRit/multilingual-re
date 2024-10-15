import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import argparse
from natsort import natsorted
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
            "example_id": example.get("example_id", idx),  # Ensure each example has a unique ID
            "prompt": prompt,
            "true_label": example["label"]
        }

def annotate_entities(text, subject_entity, object_entity):
    """
    Wrap the subject and object entities in the text with <e1> and <e2> tags.

    Args:
        text (str): Original sentence.
        subject_entity (dict): Dictionary containing 'surfaceform', 'start', 'end' for the subject.
        object_entity (dict): Dictionary containing 'surfaceform', 'start', 'end' for the object.

    Returns:
        str: Annotated sentence.
    """
    # Sort entities by start position to handle replacements correctly
    entities = sorted([('e1', subject_entity), ('e2', object_entity)], key=lambda x: x[1]['start'], reverse=True)
    
    for tag, entity in entities:
        start = entity['start']
        end = entity['end']
        surface = entity['surfaceform']
        
        # Verify that the surface form matches the text slice
        if text[start:end] != surface:
            print(f"Warning: Surface form mismatch for {tag} entity. Expected '{surface}', found '{text[start:end]}'.")
        
        # Insert the closing tag first to not mess up the indices
        text = text[:end] + f"</{tag}>" + text[end:]
        text = text[:start] + f"<{tag}>" + text[start:]
    
    return text

def construct_prompt_single_choice(example, labels):

    label_text = ""
    for key, value in labels.items():
        label_text += f"{key}: {value}\n"

    prompt = f'''Given the sentence: "{example['annotated_sent']}", which one of the following relations between the two entities <e1> and <e2> is being discussed?\nChoose one from this list of {len(labels)} options:\n{label_text}\nThe answer is: '''
    
    return prompt

def construct_prompt_dependency_choice(example, labels):
    # Create a comma-separated list of labels
    label_text = ""
    for key, value in labels.items():
        label_text += f"{key}: {value}\n"

    dependency_list = example.get("dep_graph", [])
    dep_graph = []
    # (node_dict[n1], rel, node_dict[n2])
    for dep in dependency_list:
        dep_graph.append({'head': dep[2], 'rel': dep[1], 'word': dep[0]})
    
    dep_text = json.dumps(dep_graph)
    
    prompt = f'''Given the sentence: "{example['annotated_sent']}", which one of the following relations between the two entities <e1> and <e2> is being discussed?\nWe also provide the dependency parse in the form of head, rel, and word: {dep_text}\nChoose one from this list of {len(labels)} options:\n{label_text}\nThe answer is: '''
    
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
    parser.add_argument('--dataset',        help='choice of dataset',                  type=str, default='indore')

    parser.add_argument("--batch_size",                                    type=int, default=8)
    parser.add_argument('--model_name',      help='name of the LL to use',            type=str, default='llama3')
    parser.add_argument('--dep_parser',      help='name of the dependency parser',    type=str, default='stanza') # 'None' if no dependency parser is used
    parser.add_argument('--split',           help='split',                             type=str, default='test')
    parser.add_argument('--mode',            help='mode',                              type=str, default='zshot')
    
    args = parser.parse_args()
    
    return args

if __name__ =='__main__':	
    args = get_args()

    if args.dep_parser.lower() != 'none':
        annot_file = f'../data/{args.dataset}/{args.src_lang}_prompt_{args.dep_parser}.json'
    else:
        annot_file = f'../data/{args.dataset}/{args.src_lang}_prompt_trankit.json'

    with open(annot_file) as f:
        data = json.load(f)

    split_data = data.get(args.split, [])

    # List of possible labels
    with open(f"../data/{args.dataset}/relation_dict.json") as f:
        relation_labels = json.load(f)

    labels= dict(natsorted(relation_labels.items()))

    # Flatten the data: one entry per relation
    flattened_data = []
    for example in split_data:
        docid = example.get("docid", example.get("id", ""))
        text = example.get("text", "")
        dep_graph = example.get("dep_graph", []) if args.dep_parser.lower() != 'none' else None
        relations = example.get("relations", [])
        entities = example.get("entities", [])

        # Create a mapping from entity URI to entity details for quick lookup
        entity_map = {entity['uri']: entity for entity in entities}

        for rel_idx, relation in enumerate(relations):
            predicate = relation.get("predicate", None)
            if predicate is None:
                continue  # Skip if no predicate

            if isinstance(predicate, int):
                if predicate < 0 or predicate >= len(relation_labels):
                    print(f"Warning: predicate index {predicate} out of range for labels.")
                    continue
                label = relation_labels[str(predicate)]
            else:
                print(f"Warning: unexpected predicate type {type(predicate)}.")
                continue

            # Retrieve subject and object entities
            subject = relation.get("subject", None)
            obj = relation.get("object", None)

            if subject is None or obj is None:
                print(f"Warning: Relation {rel_idx} in docid {docid} lacks subject or object.")
                continue

            # Optionally, you can resolve entities using entity_map if needed
            # For now, we'll assume that 'subject' and 'object' contain the necessary fields

            # Annotate the text with <e1> and <e2> tags
            annotated_text = annotate_entities(text, subject, obj)

            # Create a unique example_id by combining docid and relation index
            example_id = f"{docid}-{rel_idx}"

            new_example = {
                "example_id": example_id,
                "annotated_sent": annotated_text,
                "dep_graph": dep_graph,
                "label": label
            }

            flattened_data.append(new_example)

    print(f"Total flattened examples: {len(flattened_data)}")

    if args.dep_parser.lower() == 'none':    
        # Create the dataset
        dataset = LabelDatasetSingleChoice(
            data=flattened_data,
            labels=labels,
            prompt_func=construct_prompt_single_choice
        )
    else:
        # Create the dataset
        dataset = LabelDatasetSingleChoice(
            data=flattened_data,
            labels=labels,
            prompt_func=construct_prompt_dependency_choice
        )

    # Define DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Map model names to Hugging Face model IDs
    model_id_mapping = {
        'llama3': "meta-llama/Meta-Llama-3-8B-Instruct",
        'gemma2': 'google/gemma-2-9b-it',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.3'
    }

    model_id = model_id_mapping.get(args.model_name.lower(), None)
    if model_id is None:
        raise ValueError(f"Unsupported model_name '{args.model_name}'. Choose from {list(model_id_mapping.keys())}.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Assign pad_token to eos_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("pad_token was not set. Using eos_token as pad_token.")

    # Load model with appropriate precision and device mapping
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        raise e

    model.eval()  # Set model to evaluation mode
    results = []
    count = 0
    for batch in tqdm(dataloader, desc="Generating Responses"):
        try:
            prompts     = batch["prompt"]
            example_ids = batch["example_id"]
            true_labels = batch["true_label"]

            # Generate responses
            generated_texts = generate_responses(prompts, model, tokenizer)

            # Map responses
            count=0
            for ex_id, gen_text, prompt in zip(example_ids, generated_texts, prompts):
                # Clean and normalize the generated text by removing the prompt
                # Assuming the model appends its answer after the prompt
                if gen_text.startswith(prompt):
                    cleaned_gen_text = gen_text[len(prompt):].strip()
                else:
                    # If the prompt isn't at the start, attempt to find and remove it
                    idx = gen_text.find(prompt)
                    if idx != -1:
                        cleaned_gen_text = gen_text[idx + len(prompt):].strip()
                    else:
                        cleaned_gen_text = gen_text.strip()
                
                curr_data = {
                    'id': ex_id,
                    'true_label': true_labels[count],
                    'gen_text': cleaned_gen_text,
                    'prompt': prompt
                }
                #print(count)
                results.append(curr_data)
                count+=1
        # import pdb; pdb.set_trace()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            continue

    print("Generation completed.")

    output_filename = f'../prompting_predictions/{args.dataset}-{args.src_lang}-{args.model_name}_zshot_prompt-{args.dep_parser}-{args.split}.json'
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filename}")
