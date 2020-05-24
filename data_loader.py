import math
import os.path
import numpy as np
import itertools
import random
from torch.utils.data import Dataset

class DataLoader():

	def __init__(self, data_dir, max_bags=200, max_s_len=120, mode='multi_label', flip=0.0):
		self.data_dir = data_dir
		self.max_bags = max_bags
		self.max_s_len = max_s_len	
		self.mode = mode
		self.flip = flip	

		def load_dict(file):
			with open(os.path.join(data_dir, file), 'r') as f:
				n_dict = len(f.readlines())			
			with open(os.path.join(data_dir, file), 'r') as f:
				dict2id = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
				id2dict = {v: k for k, v in dict2id.items()}
			return n_dict, dict2id, id2dict	

		self.n_entity, self.entity2id, self.id2entity = load_dict('entity2id.txt')	
		print "number of entities: %d" % self.n_entity

		self.n_relation, self.relation2id, self.id2relation = load_dict('relation2id.txt')	
		print "number of relations: %d" % self.n_relation

		self.n_word, self.word2id, self.id2word = load_dict('word2id.txt')
		print "number of words: %d" % self.n_word	

		self.max_pos = 100
		self.n_pos = self.max_pos*2+1
		train_bags, train_musk, train_pos1, train_pos2, train_labels = self.get_bags(self.create_bag('train.txt', mode))

		if flip!=0:	
			flip_count = 0
			if not os.path.exists(os.path.join(data_dir, 'flip_%f.npz'%flip)):
				origin_labels = train_labels[:]
				flip_labels = []
				for i in range(len(train_labels)):
					origin = set(list(train_labels[i]))
					fliped = []
					for j in range(self.n_relation):
						if j not in origin:
							if np.random.binomial(1, flip)==1:
								flip_count += 1
								fliped.append(j)
						else:
							if np.random.binomial(1, flip)==0:
								flip_count += 1
								fliped.append(j)
					flip_labels.append(np.asarray(fliped, dtype=np.int_))
				train_labels = flip_labels
				np.savez(os.path.join(data_dir, 'flip_%f'%flip), origin_labels=origin_labels, flip_labels=flip_labels, flip_count=flip_count)
			else:
				flip_load = np.load(os.path.join(data_dir, 'flip_%f.npz'%flip))
				train_labels = flip_load['flip_labels']
				flip_count = flip_load['flip_count']
			print 'flip ratio', flip, 'average flip labels per bag', float(flip_count)/len(train_labels)


		self.n_train = len(train_bags)		
		index_list = range(self.n_train)
		random.Random(111).shuffle(index_list)
		train_select = index_list

		self.train_bags, self.train_musk, self.train_pos1, self.train_pos2, self.train_labels = [train_bags[x] for x in train_select], [train_musk[x] for x in train_select], [train_pos1[x] for x in train_select], [train_pos2[x] for x in train_select], [train_labels[x] for x in train_select]
		self.test_manual_bags, self.test_manual_musk, self.test_manual_pos1, self.test_manual_pos2, self.test_manual_labels = self.get_bags(self.create_manual_bag('manualTest.txt'))					

	def get_word2count(self):
		worddict = dict()
		with open(self.data_dir+'train.txt') as f:
			for line_ in f:
				line = line_.strip().split('	')
				words = line[5].split(' ')
				for word in words:
					if word not in worddict:
						worddict[word] = 0
					worddict[word] += 1
		with open(self.data_dir+'test.txt') as f:
			for line_ in f:
				line = line_.strip().split('	')
				words = line[5].split(' ')
				for word in words:
					if word not in worddict:
						worddict[word] = 0
					worddict[word] += 1		
		sort_word = sorted(worddict.items(), key=lambda x:x[1], reverse=True)
		with open(self.data_dir+'wordcount.txt', 'ab') as f:
			for item in sort_word:
				f.write('	'.join([item[0], str(item[1])])+'\n')

	def read_pre_train_embedding(self):
		vec = {}
		with open(self.data_dir+'vec.txt') as f:
			f.readline()
			for line_ in f.readlines():
				line = line_.strip().split(' ')
				if line[0]!=' ':
					vec[line[0]] = np.asarray([float(x) for x in line[1:]], dtype=np.float32)
		self.pre_w = vec

	def get_word2id(self, low_freq, high_freq):
		index = 0
		embed = []
		w2id = open(self.data_dir+'word2id.txt', 'ab')
		with open(self.data_dir+'wordcount.txt') as f:
			for line_ in f:
				line = line_.strip().split('	')
				if int(line[1])>=low_freq and int(line[1])<=high_freq:				
					if line[0] in self.pre_w:
						embed.append(self.pre_w[line[0]])
						w2id.write('	'.join([line[0], str(index)])+'\n')
						index += 1
		embed.append(np.random.rand(50))
		embed.append(np.zeros(50))
		np.save(self.data_dir+'word_embed.npy', np.asarray(embed, dtype=np.float32))

	def pos_embed(self, x):
		return max(0, min(x + self.max_pos, self.n_pos))

	def create_bag(self, datafile, mode='multi_class'):
		name_dict = dict()
		bags = []
		bag_index = 0
		with open(self.data_dir+datafile) as f:
			for line_ in f:
				line = line_.strip().split('	')	
				e1_id = line[0]
				e2_id = line[1]
				e1_name = line[2]
				e2_name = line[3]
				rel = line[4]
				sent = line[5].split(' ')[:-1]
				if mode=='multi_class':
					bag_name = '	'.join([e1_id, e2_id, rel])
				elif mode=='multi_label':
					bag_name = '	'.join([e1_id, e2_id])					
				s = []
				pos1 = 0
				pos2 = 0
				index = 0
				for word in sent:
					if word == e1_name:
						pos1 = index
					if word == e2_name:
						pos2 = index
					if word in self.word2id:
						s.append(self.word2id[word])
					else:
						s.append(self.n_word)
					index += 1
				if bag_name not in name_dict:
					name_dict[bag_name] = bag_index
					bag_index += 1
					bags.append([[], set()])
				bags[name_dict[bag_name]][0].append((s, pos1, pos2))	
				bags[name_dict[bag_name]][1].add(rel)	
		return bags	

	def create_manual_bag(self, datafile):
		bags = []
		with open(self.data_dir+datafile) as f:
			while 1:
				line = f.readline()
				if line!='':
					lines = line.strip().split('	')
					e1_id = lines[0]
					e2_id = lines[1]
					e1_name = lines[2]
					e2_name = lines[3]
					bag_name = '	'.join([e1_id, e2_id])
					rels = f.readline().strip().split('	')
					new_bag = [[], set(rels)]
					num_sents = int(f.readline().strip())
					for i in range(num_sents):
						sent = f.readline().strip().split(' ')
						s = []
						pos1 = 0
						pos2 = 0
						index = 0
						for word in sent:
							if word == e1_name:
								pos1 = index
							if word == e2_name:
								pos2 = index
							if word in self.word2id:
								s.append(self.word2id[word])
							else:
								s.append(self.n_word)
							index += 1
						new_bag[0].append((s, pos1, pos2))
					bags.append(new_bag)
					f.readline()
				else:
					break
		return bags

	def get_bags(self, bags):		
		normal_bags = []
		musk_idxs = []
		pos1_bags = []
		pos2_bags = []
		bag_labels = []
		for key in range(len(bags)):			
			sents = bags[key][0]
			bag_size = len(sents)
			start = 0
			while start<bag_size:
				bs = []
				musk_bs = []
				p1s = []	
				p2s = []
				#cut big bags into small ones			
				if start+self.max_bags>=bag_size:
					ss = sents[start:]
				else:
					ss = sents[start:start+self.max_bags]
				for s in ss:
					sent = s[0]
					pos = [s[1], s[2]]
					pos.sort()
					m_bs = []
					for i in range(self.max_s_len):
						if i >= len(sent):
							m_bs.append(0)
						elif i - pos[0]<=0:
							m_bs.append(1)
						elif i - pos[1]<=0:
							m_bs.append(2)
						else:
							m_bs.append(3)
					musk_bs.append(m_bs)
					p1s.append([self.pos_embed(i - s[1]) for i in range(self.max_s_len)])
					p2s.append([self.pos_embed(i - s[2]) for i in range(self.max_s_len)])
					if len(sent)>=self.max_s_len:
						exs = sent[:self.max_s_len]
					else:
						exs = sent
						exs.extend([self.n_word+1]*(self.max_s_len-len(sent)))
					exs = np.asarray(exs, dtype=np.int32)
					bs.append(exs) 				
				normal_bags.append(np.asarray(bs, dtype=np.int_)) # Sizes of tensors must match except in dimension 0 in each example in a batch
				musk_idxs.append(np.asarray(musk_bs, dtype=np.int_))
				pos1_bags.append(np.asarray(p1s, dtype=np.int_)) 
				pos2_bags.append(np.asarray(p2s, dtype=np.int_)) 				
				labels = []
				for l in bags[key][1]:
					if l in self.relation2id:
						labels.append(self.relation2id[l])
					else:
						labels.append(self.relation2id['NA'])				
				bag_labels.append(np.asarray(labels, dtype=np.int_))
				start = start+self.max_bags
		return normal_bags, musk_idxs, pos1_bags, pos2_bags, bag_labels

class RE_Dataset(Dataset):

	def __init__(self, data_loader, dataset='train', shuffle=False):

		self.data_loader = data_loader
		self.dataset = dataset
		self.shuffle = shuffle
		if dataset=='train':
			bags, musk, pos1, pos2, pos_labels = self.data_loader.train_bags, self.data_loader.train_musk, self.data_loader.train_pos1, self.data_loader.train_pos2, self.data_loader.train_labels
		elif dataset=='test':
			bags, musk, pos1, pos2, pos_labels = self.data_loader.test_manual_bags, self.data_loader.test_manual_musk, self.data_loader.test_manual_pos1, self.data_loader.test_manual_pos2, self.data_loader.test_manual_labels	
		labels = []
		for ls in pos_labels:
			label_rep = np.zeros(self.data_loader.n_relation, dtype=np.int_)
			label_rep[ls] = 1.
			labels.append(label_rep)	
		self.index = range(len(bags))
		if shuffle:
			random.shuffle(self.index)
		self.bags = [bags[x] for x in self.index]
		self.musk = [musk[x] for x in self.index]				
		self.pos1 = [pos1[x] for x in self.index]
		self.pos2 = [pos2[x] for x in self.index]
		self.labels = [labels[x] for x in self.index]

	def data_collate(self, batch):	
		X = []
		musk_idxs = []
		p1 = []
		p2 = []		
		y = []	
		i = []
		for item in batch:
			X.append(item[0])
			musk_idxs.append(item[1])
			p1.append(item[2]) 
			p2.append(item[3])						
			y.append(item[4])
			i.append(item[5])
		return [X, musk_idxs, p1, p2, y, i]					

	def __len__(self):
		return len(self.bags)

	def __getitem__(self, idx):
		X = self.bags[idx]
		musk_idxs = self.musk[idx]
		p1 = self.pos1[idx]
		p2 = self.pos2[idx]
		y = self.labels[idx]
		i = self.index[idx]
		return X, musk_idxs, p1, p2, y, i

if __name__=='__main__':
	data_loader = DataLoader('data/', flip=0.)
