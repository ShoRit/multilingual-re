from helper import *
import torch, torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter as Param
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import FastRGCNConv, RGCNConv, RGATConv

class GNNDropout(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.drop = nn.Dropout(params.drop)
	
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.drop(x), edge_index, edge_type


class GNNReLu(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.relu = nn.ReLU()
  
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.relu(x), edge_index, edge_type


class GNNRGCNConv(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		
		if self.params.gnn_model == 'rgcn':
			self.gnn 	= RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_deps)
		elif self.params.gnn_model == 'rgat':
			self.gnn 	= RGATConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_deps)
  
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.gnn(x, edge_index, edge_type), edge_index, edge_type

    
class GNNRGCNElConv(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		
		# self.gnn 	= RGCNConv(in_channels= self.params.ent_emb_dim, out_channels=self.params.ent_emb_dim, num_relations=self.params.wiki_rels)
		if self.params.gnn_model == 'rgcn':
			self.gnn 	= RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		elif self.params.gnn_model == 'rgat':
			self.gnn 	= RGATConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
  
  
	def forward(self, inp):
		x, edge_index, edge_type = inp
		return self.gnn(x, edge_index, edge_type), edge_index, edge_type

    


class DeepNet(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.rgcn_layers = []
		for i in range(self.params.gnn_depth-1):
			self.rgcn_layers.append(GNNRGCNConv(self.params))
			self.rgcn_layers.append(GNNReLu(self.params))
			self.rgcn_layers.append(GNNDropout(self.params))

		self.rgcn_layers.append(GNNRGCNConv(self.params))
		self.rgcn_layers.append(GNNReLu(self.params))
		self.rgcn_module = nn.Sequential(*self.rgcn_layers)
  
		# self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		# self.drop  = nn.Dropout(self.params.drop)
		# self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		# for id, _ in enumerate(self.rgcn_layers):
		# 	if id%2 == 0:
		# 		x = F.relu(self.rgcn_layers[id](x, edge_index, edge_type))
		# 	else:
		# 		x = self.rgcn_layers[id](x)
		x,edge_index, edge_type = self.rgcn_module((x, edge_index, edge_type))
		# x = F.relu(self.conv1(x, edge_index, edge_type))
		# x = self.drop(x)
		# x = F.relu(self.conv2(x, edge_index, edge_type))
		return x

class Net(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		self.drop  = nn.Dropout(self.params.drop)
		self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class Net5(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		self.drop  = nn.Dropout(self.params.drop)
		self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class Net7(torch.nn.Module):
	def __init__(self, params):
		super().__init__()
		self.params = params
		self.conv1 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		self.drop  = nn.Dropout(self.params.drop)
		self.conv2 = RGCNConv(in_channels= self.params.node_emb_dim, out_channels=self.params.node_emb_dim, num_relations=self.params.num_rels)
		

	def forward(self, x, edge_index, edge_type):
		x = F.relu(self.conv1(x, edge_index, edge_type))
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index, edge_type))
		return x



class MLRelResidualClassifier(nn.Module):
	def __init__(self, params):
		super().__init__()
		self.p					= params
		self.model 				= AutoModel.from_pretrained(params.ml_model)
		self.dropout			= nn.Dropout(self.p.drop)
		
		rel_in 	        		= self.model.config.hidden_size *3
  
		if self.p.dep 			=='1':
			self.rgcn    		= DeepNet(params)
			
		self.rel_classifier		= nn.Linear(rel_in, params.num_rels)
		
  
	def extract_entity(self, sequence_output, e_mask):
		extended_e_mask = e_mask.unsqueeze(1)
		extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
		return extended_e_mask.float()
	
  	
	def forward(self, bat):

		tokens_tensors 					= bat['tokens_tensors']
		segments_tensors 				= bat['segments_tensors']
		e1_mask 						= bat['marked_e1']
		e2_mask 						= bat['marked_e2']
		masks_tensors 					= bat['masks_tensors']
		label_ids 						= bat['label_ids']
		dep_data 						= bat['dep_tensors']


		# tokens_tensors, segments_tensors, e1_mask, e2_mask, masks_tensors, relation_emb, label_ids, et_1, et_2, dep_data, ent1_emb, ent2_emb, ent_data = bat
  
		bert_embs 						= self.model(input_ids=tokens_tensors,attention_mask= masks_tensors, token_type_ids= segments_tensors)
		sequence_output 				= bert_embs[0] # Sequence of hidden-states of the last layer.
		pooled_output   				= bert_embs[1] # Last layer hidden-state of the [CLS] token further processed 
		context 						= self.dropout(pooled_output)
  	
		e1_h 							= self.extract_entity(sequence_output, e1_mask)
		e2_h							= self.extract_entity(sequence_output, e2_mask)

		if self.p.dep == '1':         
			# graph_embs                = self.rgcn(dep_data.x, dep_data.edge_index, dep_data.edge_type)
			n1_mask, n2_mask, batch     = dep_data.n1_mask, dep_data.n2_mask, dep_data.batch
			batch_np 					= batch.cpu().detach().numpy()
			graph_embs 					= []
			for idx in range(0, sequence_output.shape[0]):
				bids					= np.where(batch_np==idx)[0]
				sid, eid 				= bids[0], bids[-1]+1
				graph_embs.append(torch.max(sequence_output[idx]+dep_data.x[sid:eid,:,None], dim=1)[0])

			graph_embs 					= torch.vstack(graph_embs)
			graph_embs                  = self.rgcn(graph_embs, dep_data.edge_index, dep_data.edge_type)
			

			e1_dep, e2_dep              = [],[]
			for idx in range(0,sequence_output.shape[0]):
				mask        			= torch.where(batch==idx, 1,0)
				m1, m2      			= mask*n1_mask, mask*n2_mask
				e1_dep.append(torch.mm(m1.unsqueeze(dim=0).float(),graph_embs))
				e2_dep.append(torch.mm(m2.unsqueeze(dim=0).float(),graph_embs))
   
			e1_dep          			= torch.cat(e1_dep, dim=0)
			e2_dep          			= torch.cat(e2_dep, dim=0)
	
			e1_h 						+= e1_dep
			e2_h 						+= e2_dep
		

		rel_output 						= [context, e1_h, e2_h]  

		rel_output 						= torch.cat(rel_output,  dim=-1)
		
		rel_output 						= torch.tanh(rel_output)
		rel_logits 		  				= self.rel_classifier(rel_output)

		return  {'rel_output': rel_output, 
	   			'rel_logits': rel_logits, 
	   			}







class MLRelConcatClassifier(nn.Module):
	def __init__(self, params):
		super().__init__()
		self.p					= params
		self.model 				= AutoModel.from_pretrained(params.ml_model)
   
		self.dropout			= nn.Dropout(self.p.drop)
		
		rel_in 	        		= self.model.config.hidden_size *3
		
		if self.p.dep 			=='1':
			self.rgcn    		= DeepNet(params)
			rel_in 	        	+= self.p.node_emb_dim*2
		
		self.rel_classifier		= nn.Linear(rel_in, params.num_rels)
  
	def extract_entity(self, sequence_output, e_mask):
		extended_e_mask = e_mask.unsqueeze(1)
		extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
		return extended_e_mask.float()
	
	def forward(self, bat):


		tokens_tensors 					= bat['tokens_tensors']
		segments_tensors 				= bat['segments_tensors']
		e1_mask 						= bat['marked_e1']
		e2_mask 						= bat['marked_e2']
		masks_tensors 					= bat['masks_tensors']
		label_ids 						= bat['label_ids']
		dep_data 						= bat['dep_tensors']
		  
		bert_embs 						= self.model(input_ids=tokens_tensors,attention_mask= masks_tensors, token_type_ids= segments_tensors)
		sequence_output 				= bert_embs[0] # Sequence of hidden-states of the last layer.
		pooled_output   				= bert_embs[1] # Last layer hidden-state of the [CLS] token further processed 
		context 						= self.dropout(pooled_output)
  	
		e1_h 							= self.extract_entity(sequence_output, e1_mask)
		e2_h							= self.extract_entity(sequence_output, e2_mask)

		rel_output 						= [context, e1_h, e2_h]
  
		if self.p.dep == '1':         
			# graph_embs                = self.rgcn(dep_data.x, dep_data.edge_index, dep_data.edge_type)
			n1_mask, n2_mask, batch     = dep_data.n1_mask, dep_data.n2_mask, dep_data.batch
			batch_np 					= batch.cpu().detach().numpy()
			graph_embs 					= []
			for idx in range(0, sequence_output.shape[0]):
				bids					= np.where(batch_np==idx)[0]
				sid, eid 				= bids[0], bids[-1]+1
				graph_embs.append(torch.max(sequence_output[idx]+dep_data.x[sid:eid,:,None], dim=1)[0])


			graph_embs 					= torch.vstack(graph_embs)
			graph_embs                  = self.rgcn(graph_embs, dep_data.edge_index, dep_data.edge_type)

			e1_dep, e2_dep              = [],[]
			for idx in range(0,sequence_output.shape[0]):
				mask        			= torch.where(batch==idx, 1,0)
				m1, m2      			= mask*n1_mask, mask*n2_mask
				e1_dep.append(torch.mm(m1.unsqueeze(dim=0).float(),graph_embs))
				e2_dep.append(torch.mm(m2.unsqueeze(dim=0).float(),graph_embs))

   
			e1_dep          			= torch.cat(e1_dep, dim=0)
			e2_dep          			= torch.cat(e2_dep, dim=0)
	
			rel_output.append(e1_dep)
			rel_output.append(e2_dep)
   
			

		rel_output 						= torch.cat(rel_output,  dim=-1)		
		rel_output 						= torch.tanh(rel_output)
		rel_logits 		  				= self.rel_classifier(rel_output)

		
		return  {'rel_output': rel_output, 
	   			'rel_logits': rel_logits, 
	   			}





