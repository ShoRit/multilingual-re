from helper import *
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_geometric.data import Data
import torch

class RelDataset(Dataset):
    def __init__(self, dataset, params, data_idx=0):
        self.dataset		= dataset
        self.p				= params
        self.tokenizer      = AutoTokenizer.from_pretrained(self.p.ml_model)
        self.data_idx 		= data_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele             = self.dataset[idx]

        '''
        Configurations of a single element in the dataset:

        'tokens'      : mlm_toks,
        'tok_range'   : tok_range,    
        'e1_ids'      : e1_toks,
        'e2_ids'      : e2_toks,
        'label'       : rel_type,
        'dep_data'    : dep_data,
        'doc_text'    : doc_text
        '''


        tokens 			= torch.tensor(ele['tokens']	+ [0]*(self.p.max_seq_len-len(ele['tokens'])))
        marked_e1		= torch.tensor(ele['e1_ids']	+ [0]*(self.p.max_seq_len-len(ele['tokens'])))
        marked_e2		= torch.tensor(ele['e2_ids']	+ [0]*(self.p.max_seq_len-len(ele['tokens'])))
        segments		= torch.tensor([0]*len(tokens))

        label 			= torch.LongTensor(ele['label'])

        dep_data 		= ele['dep_data']
        dep_data 		= Data(x=torch.tensor(dep_data.x), edge_index= torch.tensor(dep_data.edge_index), 
                              edge_type= torch.tensor(dep_data.edge_type), n1_mask=torch.tensor(dep_data.n1_mask), n2_mask=torch.tensor(dep_data.n2_mask))    

        return 	{
                    'tokens': tokens,
                    'segments': segments,
                    'marked_e1': marked_e1,
                    'marked_e2': marked_e2,
                    'label': label,
                    'dep_data': dep_data
        }


def create_mini_batch(samples):
    import torch
    from torch_geometric.loader import DataLoader 
    tokens_tensors 					= [s['tokens'] for s in samples]
    segments_tensors 				= [s['segments'] for s in samples]
    marked_e1 						= [s['marked_e1'] for s in samples]
    marked_e2 						= [s['marked_e2'] for s in samples]

    label_ids 						= torch.stack([s['label'] for s in samples])


    tokens_tensors 					= pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors 				= pad_sequence(segments_tensors, batch_first=True)
    marked_e1 						= pad_sequence(marked_e1, batch_first=True)
    marked_e2 						= pad_sequence(marked_e2, batch_first=True)
    masks_tensors 					= torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors 					= masks_tensors.masked_fill(tokens_tensors != 0, 1)

    dep_list 						= [s['dep_data'] for s in samples]
    dep_loader 					    = DataLoader(dep_list, batch_size=len(dep_list))
    dep_tensors 					= [elem for elem in  dep_loader][0]


    return {
                'tokens_tensors': tokens_tensors, 
                'segments_tensors': segments_tensors,
                'marked_e1': marked_e1,
                'marked_e2': marked_e2,
                'masks_tensors': masks_tensors,
                'label_ids': label_ids,
                'dep_tensors': dep_tensors,    
            }


    # return tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, label_ids, et_1, et_2, graph_tensors, ent1_emb, ent2_emb, ent_tensors


