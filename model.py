import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available() 	
data_dir = 'data/'	

class Embedding(nn.Module):
	'''
	position embedding and word embedding
	'''
	def __init__(self, n_word, n_pos, input_size, pos_size, position=False, pretrain=True):
		super(Embedding, self).__init__()	
		self.n_word = n_word
		self.n_pos = n_pos		
		self.input_size = input_size   
		self.pos_size = pos_size 	
		self.position = position
		self.pretrain = pretrain	     
		self.embedding = nn.Embedding(n_word+2, input_size, padding_idx=n_word+1)
		if pretrain:
			self.embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.load(data_dir+'word_embed.npy'), dtype=np.float32)))		
		if position:
			self.pos1_embedding = nn.Embedding(n_pos+1, pos_size, padding_idx=n_pos)
			self.pos1_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(91).uniform(low=-0.1, high=0.1, size=(n_pos+1, pos_size)), dtype=np.float32)))			
			self.pos2_embedding = nn.Embedding(n_pos+1, pos_size, padding_idx=n_pos)			
			self.pos2_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(92).uniform(low=-0.1, high=0.1, size=(n_pos+1, pos_size)), dtype=np.float32)))			

	def forward(self, inputs, pos1, pos2):
		embedded = self.embedding(inputs)
		if self.position:
			p1_embed = self.pos1_embedding(pos1)
			p2_embed = self.pos2_embedding(pos2)
			embedded = torch.cat([embedded, p1_embed, p2_embed], 2)			
		return embedded

class CNNEncoder(nn.Module):
	'''
	CNN sentence encoder
	'''
	def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, dropout=0.5, window=3, position=False, pretrain=True):
		super(CNNEncoder, self).__init__()	
		self.n_word = n_word
		self.n_pos = n_pos		
		self.input_size = input_size   
		self.pos_size = pos_size 		     
		self.hidden_size = hidden_size
		self.window = window
		self.dropout = nn.Dropout(p=dropout)		
		self.position = position
		self.pretrain = pretrain
		self.embedding = Embedding(n_word, n_pos, input_size, pos_size, position, pretrain)				
		self.conv2d = nn.Conv2d(input_size+pos_size*2, hidden_size, (1, window), padding=(0, 1))
		self.conv2d.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(31).uniform(low=-0.1, high=0.1, size=(hidden_size, input_size+pos_size*2, 1, window)), dtype=np.float32)))
		self.conv2d.bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(hidden_size), dtype=np.float32)))		

	def forward(self, inputs, musk, pos1, pos2):
		embedded = self.embedding(inputs, pos1, pos2)		
		embedded=embedded.transpose(1, 2).unsqueeze(2) # batch*in_channels*in_height*in_width
		conv = self.conv2d(embedded)
		conv = conv.squeeze(2).transpose(1, 2)
		pooled = torch.max(conv, dim=1)[0]
		activated = F.relu(pooled)
		output = self.dropout(activated)			
		return output

class PCNNEncoder(nn.Module):
	'''
	PCNN sentence encoder
	'''
	def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, dropout=0.5, window=3, position=False, pretrain=True, max_pos=100):
		super(PCNNEncoder, self).__init__()	
		self.n_word = n_word
		self.n_pos = n_pos		
		self.input_size = input_size   
		self.pos_size = pos_size 		     
		self.hidden_size = hidden_size
		self.window = window
		self.dropout = nn.Dropout(p=dropout)
		self.position = position
		self.pretrain = pretrain
		self.max_pos = max_pos

		self.embedding = Embedding(n_word, n_pos, input_size, pos_size, position, pretrain)

		self.musk_embedding = nn.Embedding(4, 3)
		self.musk_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)))	
		self.musk_embedding.weight.requires_grad = False	

		self.conv2d = nn.Conv2d(input_size+pos_size*2, hidden_size, (1, window), padding=(0, 1))
		self.conv2d.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(31).uniform(low=-0.1, high=0.1, size=(hidden_size, input_size+pos_size*2, 1, window)), dtype=np.float32)))
		self.conv2d.bias = nn.Parameter(torch.from_numpy(np.asarray(np.zeros(hidden_size), dtype=np.float32)))			

	def forward(self, inputs, musk, pos1, pos2):
		# inputs: bag*seq_len
		# musk: bag*seq_len
		poolsize = inputs.size()[1]	
		embedded = self.embedding(inputs, pos1, pos2)			
		embedded=embedded.transpose(1, 2).unsqueeze(2) # batch*in_channels*in_height*in_width
		conv = self.conv2d(embedded)
		conv = conv.squeeze(2).transpose(1, 2).unsqueeze(3)
		pooled = torch.max(conv+self.musk_embedding(musk).view(-1, poolsize, 1, 3)*self.max_pos, dim=1)[0]-self.max_pos
		activated = F.relu(pooled.view(-1, self.hidden_size * 3))
		output = self.dropout(activated)			
		return output

class GRUEncoder(nn.Module):
	'''
	GRU sentence encoder
	'''
	def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, dropout=0.5, position=False, bidirectional=True, pretrain=True):
		super(GRUEncoder, self).__init__()
		self.n_word = n_word
		self.n_pos = n_pos		
		self.input_size = input_size   
		self.pos_size = pos_size 		     
		self.hidden_size = hidden_size
		self.position = position
		self.bidirectional = bidirectional
		self.pretrain = pretrain
		self.embedding = Embedding(n_word, n_pos, input_size, pos_size, position, pretrain)		
		self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=bidirectional)

	def forward(self, inputs, musk, pos1, pos2):
		embedded = self.embedding(inputs, pos1, pos2)
		output, hidden = self.gru(embedded)
		output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
		weight = F.softmax(torch.matmul(output, self.W), 1)
		return torch.sum(weight*output, 1)

class BagEncoder(nn.Module):
	def __init__(self, n_word, n_pos, input_size, pos_size, hidden_size, position=False, encode_model='BiGRU', dropout=0.5):
		super(BagEncoder, self).__init__()
		self.n_word = n_word	
		self.n_pos = n_pos				
		self.input_size = input_size 
		self.pos_size = pos_size		        
		self.hidden_size = hidden_size
		self.position = position
		self.encode_model = encode_model
		self.dropout = dropout	
		if self.encode_model=='BiGRU':	
			self.encoder = GRUEncoder(n_word, n_pos, input_size, pos_size, hidden_size, dropout=dropout, position=position)
		elif self.encode_model=='CNN':
			self.encoder = CNNEncoder(n_word, n_pos, input_size, pos_size, hidden_size, dropout=dropout, position=position)
		elif self.encode_model=='PCNN':
			self.encoder = PCNNEncoder(n_word, n_pos, input_size, pos_size, hidden_size, dropout=dropout, position=position)

	def forward(self, inputs, musk, pos1, pos2):
		# inputs: [bag1*seq_len, bag2*seq_len, ......]		
		out_puts = []
		for i in range(len(inputs)):
			out_puts.append(self.encoder(inputs[i], musk[i], pos1[i], pos2[i]))
		return out_puts #[bag1*h, bag2*h, ...]

class Extractor(nn.Module):
	'''
	sentence selector
	'''
	def __init__(self, input_size, n_class, reduce_method='multir_att', use_bias=True):
		super(Extractor, self).__init__()
		self.input_size = input_size 
		self.n_class = n_class
		self.reduce_method = reduce_method
		self.use_bias = use_bias
		self.label_embedding = nn.Embedding(n_class, input_size)	
		self.label_embedding.weight = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(201).uniform(low=-0.1, high=0.1, size=(n_class, input_size)), dtype=np.float32)))
		self.A = nn.Parameter(torch.from_numpy(np.asarray(np.random.RandomState(211).uniform(low=-0.1, high=0.1, size=(input_size)), dtype=np.float32)))
		if self.use_bias:
			self.bias = nn.Parameter(torch.from_numpy(np.zeros((1, n_class), dtype=np.float32)))

	# multi-relation attention
	def multir_att(self, sent_embeddings):	
		label_embeddings = self.label_embedding.weight
		score = torch.matmul(sent_embeddings*self.A, label_embeddings.t())		
		weight = F.softmax(score.t(), 1)
		reduce_bag = torch.matmul(weight, sent_embeddings)
		scores = torch.sum(reduce_bag*label_embeddings, 1, keepdim=True)
		if self.use_bias:
			scores = scores.t() + self.bias
		return scores				

	def mean(self, sent_embeddings):
		label_embeddings = self.label_embedding.weight
		reduce_bag = torch.mean(sent_embeddings, 0, keepdim=True)
		scores = torch.matmul(reduce_bag, label_embeddings.t())
		if self.use_bias:
			scores = scores + self.bias				
		return scores	

	def cross_max(self, sent_embeddings):
		label_embeddings = self.label_embedding.weight
		reduce_bag = torch.max(sent_embeddings, 0)[0]
		scores = torch.matmul(reduce_bag, label_embeddings.t())
		if self.use_bias:
			scores = scores + self.bias				
		return scores

	def forward(self, inputs):
		batch_scores = []	
		# labels = torch.unbind(labels, 0)
		for i in range(len(inputs)):
			# Sentence embedding in a bag
			sent_embeddings = inputs[i]
			if self.reduce_method=='multir_att':
				scores = self.multir_att(sent_embeddings)							
			# Cross-sentence Max-musking
			elif self.reduce_method=='cross_max':
				scores = self.cross_max(sent_embeddings)				
			elif self.reduce_method=='mean':
				scores = self.mean(sent_embeddings)	
			batch_scores.append(scores)						
		return torch.cat(batch_scores, 0)

	def pred(self, inputs):
		scores_all = []
		for i in range(len(inputs)):
			# Sentence embedding in a bag
			sent_embeddings = inputs[i]	
			if self.reduce_method=='multir_att':
				scores = self.multir_att(sent_embeddings)									
			# Cross-sentence Max-musking
			elif self.reduce_method=='cross_max':
				scores = self.cross_max(sent_embeddings)
			elif self.reduce_method=='mean':
				scores = self.mean(sent_embeddings)
			scores_all.append(scores)																	
		return torch.cat(scores_all, 0)	# batchs * dims


class Z_Y(nn.Module):
	"""nEM transition module: P(Z|Y)"""
	def __init__(self, n_class, init1=1., init2=0., na_init1=1., na_init2=0.):
		# init1=p(z=1|y=1), init2=p(z=1|y=0)
		super(Z_Y, self).__init__()
		self.n_class = n_class
		self.init1 = init1
		self.init2 = init2
		self.na_init1 = na_init1
		self.na_init2 = na_init2
		self.phi = nn.Embedding(n_class, 2)
		temp = np.tile(np.asarray([init1, init2], dtype=np.float32).reshape(1, 2), (n_class, 1))
		temp[0, 0] = na_init1
		temp[0, 1] = na_init2
		self.phi.weight = nn.Parameter(torch.from_numpy(temp))
		self.phi.weight.requires_grad = False				
		self.mask_z = nn.Embedding(2, 2)
		self.mask_z.weight = nn.Parameter(torch.from_numpy(np.asarray([[0., 1.],[1., 0.]], dtype=np.float32)))
		self.mask_z.weight.requires_grad = False		
	def forward(self, z):	
		z_y = self.phi.weight
		_z_y = 1. - z_y
		z_y_ = torch.cat([z_y, _z_y], 1).view(-1, 2, 2)
		musk = self.mask_z(z).unsqueeze(2)
		z_y = torch.matmul(musk, z_y_).squeeze(2)		
		return z_y

class Y_S(nn.Module):
	"""nEM prediction module: P(Y|S)"""
	def __init__(self, embed_dim, pos_dim, hidden_dim, n_word, n_pos, n_class, reg_weight, reduce_method='mean', position=True, encode_model='PCNN', dropout=0.5, sigmoid=False):
		super(Y_S, self).__init__()
		self.embed_dim = embed_dim
		self.pos_dim = pos_dim		
		self.hidden_dim = hidden_dim		
		self.n_word = n_word
		self.n_pos = n_pos						
		self.n_class = n_class		
		self.reg_weight = reg_weight
		self.reduce_method = reduce_method
		self.position = position	
		self.encode_model = encode_model
		self.dropout = dropout
		self.sigmoid = sigmoid
		self.bag_encoder = BagEncoder(n_word, n_pos, embed_dim, pos_dim, hidden_dim, position, encode_model, dropout)
		if encode_model=='PCNN':
			hidden_dim = hidden_dim*3			
		self.extractor = Extractor(hidden_dim, n_class, reduce_method)							

	def forward(self, bags, musk_idxs, pos1, pos2):
		groups = self.bag_encoder(bags, musk_idxs, pos1, pos2)
		scores = self.extractor(groups)	
		if self.sigmoid:			
			y1_s = torch.sigmoid(scores)
		else:	
			y1_s = F.softmax(scores, 1)		 
		return y1_s

class RE(nn.Module):
	"""traning nEM"""
	def __init__(self, embed_dim, pos_dim, hidden_dim, n_word, n_pos, n_class, reg_weight, reduce_method='mean', position=True, encode_model='PCNN', dropout=0.5, sigmoid=False, init1=0., init2=0., na_init1=10., na_init2=-10.):
		super(RE, self).__init__()
		self.embed_dim = embed_dim
		self.pos_dim = pos_dim		
		self.hidden_dim = hidden_dim		
		self.n_word = n_word
		self.n_pos = n_pos						
		self.n_class = n_class		
		self.reg_weight = reg_weight
		self.reduce_method = reduce_method
		self.position = position	
		self.encode_model = encode_model
		self.dropout = dropout
		self.sigmoid = sigmoid	
		self.init1 = init1
		self.init2 = init2	
		self.na_init1 = na_init1
		self.na_init2 = na_init2
		self.y_s = Y_S(embed_dim, pos_dim, hidden_dim, n_word, n_pos, n_class, reg_weight, reduce_method, position, encode_model, dropout, sigmoid)
		self.z_y = Z_Y(n_class, init1, init2, na_init1, na_init2)	

	def reg_loss(self):
		reg_c = torch.norm(self.y_s.bag_encoder.encoder.embedding.embedding.weight) + torch.norm(self.y_s.extractor.label_embedding.weight) + torch.norm(self.y_s.extractor.A)
		if self.position:
			reg_c = reg_c + torch.norm(self.y_s.bag_encoder.encoder.embedding.pos1_embedding.weight) + torch.norm(self.y_s.bag_encoder.encoder.embedding.pos2_embedding.weight)		
		if self.encode_model=='BiGRU':
			reg_c = reg_c + torch.norm(self.y_s.bag_encoder.encoder.W)
		elif self.encode_model=='CNN' or self.encode_model=='PCNN':
			reg_c = reg_c + torch.norm(self.y_s.bag_encoder.encoder.conv2d.weight)

		return reg_c	

	def baseModel(self, bags, musk_idxs, pos1, pos2, labels):
		pred = self.y_s(bags, musk_idxs, pos1, pos2)
		output = torch.max(pred, dim=1)[1]	
		pred = torch.clamp(pred, 1e-4, 1.0-1e-4)		
		y = labels.to(torch.float)
		if self.sigmoid:
			sum_i = torch.sum(y*torch.log(pred)+(1.-y)*torch.log(1.-pred), 1)
			loss = -torch.mean(sum_i) 
		else:			
			sum_i = torch.sum(y*torch.log(pred), 1)
			loss = -torch.mean(sum_i) 
		
		return loss+self.reg_weight*self.reg_loss()								

	# e_step: computing Q(Y)
	def E_step(self, bags, musk_idxs, pos1, pos2, labels):
		y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
		y0_s = 1. - y1_s
		y_s = torch.cat([y1_s.unsqueeze(2), y0_s.unsqueeze(2)], 2)
		z_y = self.z_y(labels)
		z_y_y_s = z_y*y_s
		z_y_y_s_ = torch.unbind(z_y_y_s, 2)[0]
		z_s = torch.sum(z_y_y_s, 2).clamp(min=1e-5)
		Q_y1 = z_y_y_s_/z_s
		return Q_y1		

	# m_step: loss function (lower bound). optimizing usiing gradient descent
	def M_step(self, bags, musk_idxs, pos1, pos2, labels, Q_y1):
		y1_s = self.y_s(bags, musk_idxs, pos1, pos2)
		y0_s = 1. - y1_s				
		y_s = torch.cat([y1_s.unsqueeze(2), y0_s.unsqueeze(2)], 2) 
		y_s = torch.clamp(y_s, 1e-5, 1.0-1e-5)		
		z_y = self.z_y(labels)
		z_y = torch.clamp(z_y, 1e-5, 1.0-1e-5)
		log_z_y_y_s = torch.log(z_y) + torch.log(y_s)
		Q_y0 = 1. - Q_y1		
		Q_y = torch.cat([Q_y1.unsqueeze(2), Q_y0.unsqueeze(2)], 2)
		sum_bit = torch.sum(Q_y*log_z_y_y_s, 2)		
		sum_i = torch.sum(sum_bit, 1)
		loss = -torch.mean(sum_i)
		return loss+self.reg_weight*self.reg_loss()

	def pred(self, bags, musk_idxs, pos1, pos2):	
		return self.y_s(bags, musk_idxs, pos1, pos2)