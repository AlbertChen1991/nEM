import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils import data
from data_loader import DataLoader, RE_Dataset
from model import RE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='RE')
parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15K_D/')
parser.add_argument('--mode', dest='mode', type=str, help="Data mode: multi_label or multi_class", default='multi_label')
parser.add_argument('--run_mode', dest='run_mode', type=str, help="train, test", default='train')
parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.1)
parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=200)
parser.add_argument("--pos_dim", dest='pos_dim', type=int, help="Embedding dimension", default=200)
parser.add_argument("--hdim", dest='hdim', type=int, help="Hidden layer dimension", default=100)
parser.add_argument("--max_bags", dest="max_bags", type=int, help="Max sentences in a bag", default=5)
parser.add_argument("--max_len", dest="max_len", type=int, help="Max description length", default=30)
parser.add_argument('--encoder', dest='encoder', type=str, help="Encoder type", default='BiGRU')
parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=10)
parser.add_argument("--test_batch", dest='test_batch', type=int, help="Test batch size", default=100)
parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
parser.add_argument("--save_dir", dest='save_dir', type=str, help="Saved model path", default='./')
parser.add_argument("--resume", dest='resume', type=int, help="Resume from epoch model file", default=-1)
parser.add_argument('--reduce_method', dest='reduce_method', type=str, help="The bag reduce methods", default='attention')
parser.add_argument('--position', dest='position', action="store_true", help="If using position embedding", default=False)
parser.add_argument("--eval_start", dest='eval_start', type=int, help="Epoch when evaluation start", default=90)
parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluation per x iteration", default=1)
parser.add_argument("--save_m", dest='save_m', type=int, help="Number of saved models", default=1)
parser.add_argument("--epochs", dest='epochs', type=int, help="Epochs to run", default=100)
parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
parser.add_argument('--reg_weight', dest='reg_weight', type=float, help="The regularization weight of autoencoder", default=1e-5)
parser.add_argument('--clip', dest='clip', type=float, help="The gradient clip max norm", default=0.0)
parser.add_argument('--init1', dest='init1', type=float, help="Initialization of p(z=1|y=1)", default=1.)
parser.add_argument('--init2', dest='init2', type=float, help="Initialization of p(z=1|y=0)", default=0.)
parser.add_argument('--na_init1', dest='na_init1', type=float, help="Initialization of p(z=1|y=1) for NA relation", default=1.)
parser.add_argument('--na_init2', dest='na_init2', type=float, help="Initialization of p(z=1|y=0) for NA relation", default=0.)
parser.add_argument('--em', dest='em', action="store_true", help="Using EM", default=False)
parser.add_argument('--sigmoid', dest='sigmoid', action="store_true", help="Using sigmoid to activate scores", default=False)
parser.add_argument("--per_e_step", dest='per_e_step', type=int, help="How many m steps before e step", default=500)
parser.add_argument('--flip', dest='flip', type=float, help="Label flip rate", default=0.)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def use_optimizer(model, lr, weight_decay=0, lr_decay=0, momentum=0, rho=0.95, method='sgd'):
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	if method=='sgd':
		return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
	elif method=='adagrad':
		return optim.Adagrad(parameters, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
	elif method=='adadelta':
		return optim.Adadelta(parameters, rho=rho, lr=lr, weight_decay=weight_decay)
	elif method=='adam':
		return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
	elif method=='rmsprop':
		return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)		
	else:
		raise Exception("Invalid method, option('sgd', 'adagrad', 'adadelta', 'adam')")

def save_checkpoint(state, step):
	torch.save(state, args.save_dir+'%d_step.mod.tar'%step)	

def resume(step):
	load_file = args.save_dir+'%d_step.mod.tar'%step
	if os.path.isfile(load_file):	
		checkpoint = torch.load(load_file)	
		return checkpoint	

def MAP(Y_pred, Y_test):
	# the mean average precision given prediction
	(n_items, n_labels) = Y_pred.shape
	sort_pred = np.argsort(Y_pred, axis=1)[:,::-1]
	ranked = []
	for i in range(0, n_items):
		rank_list = []
		for j in range(0, n_labels):
			if Y_test[i][sort_pred[i][j]]==1:
				rank_list.append(j+1)
		ranked.append(rank_list)
	AP = []
	for item in ranked:
		if len(item)!=0:
			AP.append(np.mean(np.asarray(range(1,len(item)+1), dtype=np.float32)/np.array(item)))
	return np.mean(np.array(AP))

def precision_recall(y_score, Y_test):
	sort_pred = np.argsort(y_score.ravel())[::-1]
	sort_y = Y_test.ravel()[sort_pred]
	all_positive = np.sum(sort_y)
	positives = 0
	precision = []
	recall = []
	for i in range(0, len(sort_y)):
		if sort_y[i]==1:
			positives += 1
		precision.append(positives/float(i+1))
		recall.append(positives/float(all_positive))
	return np.asarray(precision, dtype=np.float32), np.asarray(recall, dtype=np.float32)

def get_pre_recall(y_score, Y_test, name='y'):
	precision, recall = precision_recall(y_score, Y_test)		
	np.save(args.save_dir+'%s_precision'%name, precision)
	np.save(args.save_dir+'%s_recall'%name, recall)

def pr_with_threshold(y_score, Y_test, threshold=0.5):
	gold = 0.
	pred = 0.
	correct = 0.
	shapes = y_score.shape
	for i in range(shapes[0]):
		for j in range(shapes[1]):
			if Y_test[i, j]==1:
				gold += 1
				if y_score[i, j]>threshold:
					pred += 1
					correct += 1
			else:
				if y_score[i, j]>threshold:
					pred += 1
	if pred==0 or gold==0:
		p = 0
		r = 0
		f1 = 0
	else:
		p = correct/pred
		r = correct/gold
		f1 = 2.0*p*r/(p+r)
	return p,r,f1		

def write_list_to_file(l, file):
	with open(file, 'wb') as f:
		strs = ','.join([str(x) for x in l])
		f.write(strs+'\n')

def PR_curve(cdir='pr'):	
	plt.clf()
	filename = [['PCNN+ATT', 'PCNN+ATT+nEM'], ['PCNN+MEAN', 'PCNN+MEAN+nEM'], ['PCNN+MAX', 'PCNN+MAX+nEM']]
	# filename = [['PCNN+ATT', 'PCNN+ATT+LD'], ['PCNN+ATT+0.01', 'PCNN+ATT+LD+0.01'], ['PCNN+ATT+0.02', 'PCNN+ATT+LD+0.02'], ['PCNN+ATT+0.04', 'PCNN+ATT+LD+0.04'], ['PCNN+ATT+0.08', 'PCNN+ATT+LD+0.08']]
	color = ['red', 'green', 'black']
	# color = ['red', 'magenta', 'darkorange', 'cornflowerblue', 'black']
	# color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'green', 'black', 'magenta']	
	for i in range(len(filename)):
		p1 = np.load('./%s/'%cdir+filename[i][0]+'_precision.npy')[:3000]
		r1  = np.load('./%s/'%cdir+filename[i][0]+'_recall.npy')[:3000]
		p2 = np.load('./%s/'%cdir+filename[i][1]+'_precision.npy')[:3000]
		r2  = np.load('./%s/'%cdir+filename[i][1]+'_recall.npy')[:3000]		
		plt.plot(r1, p1, color = color[i], ls=':', lw=1.5, label = filename[i][0])
		plt.plot(r2, p2, color = color[i], ls='-', lw=2, label = filename[i][1])
		write_list_to_file(p1, 'drawcurve/pr/'+filename[i][0]+'_precision.txt')
		write_list_to_file(r1, 'drawcurve/pr/'+filename[i][0]+'_recall.txt')
		write_list_to_file(p2, 'drawcurve/pr/'+filename[i][1]+'_precision.txt')
		write_list_to_file(r2, 'drawcurve/pr/'+filename[i][1]+'_recall.txt')		
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.6, 0.9])
	plt.xlim([0.0, 0.5])
	plt.legend(loc="upper right")
	leg = plt.gca().get_legend()
	ltext = leg.get_texts()
	plt.setp(ltext, fontsize='small')	
	plt.grid(True)	 
	plt.savefig('%s/PR_curve.png'%cdir)	

def pred_RE(model, data_loader, dataset='test', name='y'):
	model.eval()	
	dataset = RE_Dataset(data_loader, dataset=dataset)
	trainloader = data.DataLoader(dataset, batch_size=args.batch, num_workers=args.n_worker, collate_fn=dataset.data_collate)
	pred_all = []
	y_all = []	
	for batch_idx, (X, musk, p1, p2, y, index) in enumerate(trainloader):	
		X, musk, p1, p2, y = [torch.from_numpy(x) for x in X], [torch.from_numpy(x) for x in musk], [torch.from_numpy(x) for x in p1], [torch.from_numpy(x) for x in p2], torch.from_numpy(np.asarray(y, dtype=np.float32))
		X, musk, p1, p2, y = [Variable(x, requires_grad=False) for x in X], [Variable(x, requires_grad=False) for x in musk], [Variable(x, requires_grad=False) for x in p1], [Variable(x, requires_grad=False) for x in p2], Variable(y, requires_grad=False)		
		if use_cuda:
			X, musk, p1, p2, y = [x.cuda() for x in X], [x.cuda() for x in musk], [x.cuda() for x in p1], [x.cuda() for x in p2], y.cuda()
		pred = model.pred(X, musk, p1, p2)
		pred = pred.data.cpu().numpy()
		y = y.data.cpu().numpy()			
		pred_all.append(pred)
		y_all.append(y) 
	y_score = np.concatenate(pred_all)
	Y_test = np.concatenate(y_all)
	ap = average_precision_score(Y_test.ravel(), y_score.ravel())
	p,r,f1 = pr_with_threshold(y_score, Y_test)
	return y_score,Y_test,ap,p,r,f1

def e_step(model, data_loader, init=False):
	dataset = RE_Dataset(data_loader, dataset='train')
	trainloader = data.DataLoader(dataset, batch_size=args.batch, num_workers=args.n_worker, collate_fn=dataset.data_collate)
	batchs = len(trainloader)	
	Q_y1_all = []
	index_all = []
	for batch_idx, (X, musk, p1, p2, y, index) in enumerate(trainloader):
		if init:
			Q_y1_all.append(y)	
		else:
			X, musk, p1, p2, y = [torch.from_numpy(x) for x in X], [torch.from_numpy(x) for x in musk], [torch.from_numpy(x) for x in p1], [torch.from_numpy(x) for x in p2], torch.from_numpy(np.asarray(y, dtype=np.int_))
			X, musk, p1, p2, y = [Variable(x) for x in X], [Variable(x) for x in musk], [Variable(x) for x in p1], [Variable(x) for x in p2], Variable(y)
			if use_cuda:
				X, musk, p1, p2, y = [x.cuda() for x in X], [x.cuda() for x in musk], [x.cuda() for x in p1], [x.cuda() for x in p2], y.cuda()
			model.eval()		
			Q_y1 = model.E_step(X, musk, p1, p2, y)
			Q_y1_all.append(Q_y1.data.cpu().numpy())
		index_all = index_all + index
	Q_y1_all = np.concatenate(Q_y1_all, 0)
	map_index = {index_all[x]:x for x in range(len(index_all))}
	list_index = [map_index[x] for x in range(len(index_all))]
	return Q_y1_all[list_index]

def em_step(model, data_loader, optimizer, epoch, step, Q_y1_all):
	dataset = RE_Dataset(data_loader, dataset='train', shuffle=True)
	trainloader = data.DataLoader(dataset, batch_size=args.batch, num_workers=args.n_worker, collate_fn=dataset.data_collate)
	batchs = len(trainloader)	
	all_step = step + batchs	
	losses = []			
	for batch_idx, (X, musk, p1, p2, y, index) in enumerate(trainloader):	
		X, musk, p1, p2, y, Q_y1 = [torch.from_numpy(x) for x in X], [torch.from_numpy(x) for x in musk], [torch.from_numpy(x) for x in p1], [torch.from_numpy(x) for x in p2], torch.from_numpy(np.asarray(y, dtype=np.int_)), torch.from_numpy(np.asarray(Q_y1_all[index], dtype=np.float32))
		X, musk, p1, p2, y, Q_y1 = [Variable(x) for x in X], [Variable(x) for x in musk], [Variable(x) for x in p1], [Variable(x) for x in p2], Variable(y), Variable(Q_y1, requires_grad=False)
		if use_cuda:
			X, musk, p1, p2, y, Q_y1 = [x.cuda() for x in X], [x.cuda() for x in musk], [x.cuda() for x in p1], [x.cuda() for x in p2], y.cuda(), Q_y1.cuda()
		model.train()
		loss = model.M_step(X, musk, p1, p2, y, Q_y1)
		loss_ = loss.item()	
		losses.append(loss_)
		optimizer.zero_grad()
		loss.backward()
		if args.clip!=0:
			nn.utils.clip_grad_norm(model.parameters(), args.clip)		
		optimizer.step()
		step = step + 1		
		print '[Epoch %d/%d ] | Iter %d/%d | Loss %f' % (epoch, args.epochs, step, all_step, loss_)	
		if step%args.eval_per==0:
			save_checkpoint({
				'epoch': epoch,
				'step': step,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict()
			}, step)
		if step%args.per_e_step==0:					
			print 'executing e step...'
			Q_y1_all = e_step(model, data_loader)	
			np.save(args.save_dir+'Q_y', Q_y1_all)						
			print 'executing m step...'	

	return np.mean(np.array(losses)), step, Q_y1_all

def train_epoch_RE(model, data_loader, optimizer, epoch, step):	
	dataset = RE_Dataset(data_loader, dataset='train', shuffle=True)
	trainloader = data.DataLoader(dataset, batch_size=args.batch, num_workers=args.n_worker, collate_fn=dataset.data_collate)
	batchs = len(trainloader)
	losses = []
	all_step = step + batchs
	for batch_idx, (X, musk, p1, p2, y, index) in enumerate(trainloader):	
		X, musk, p1, p2, y = [torch.from_numpy(x) for x in X], [torch.from_numpy(x) for x in musk], [torch.from_numpy(x) for x in p1], [torch.from_numpy(x) for x in p2], torch.from_numpy(np.asarray(y, dtype=np.int_))
		X, musk, p1, p2, y = [Variable(x) for x in X], [Variable(x) for x in musk], [Variable(x) for x in p1], [Variable(x) for x in p2], Variable(y)
		if use_cuda:
			X, musk, p1, p2, y= [x.cuda() for x in X], [x.cuda() for x in musk], [x.cuda() for x in p1], [x.cuda() for x in p2], y.cuda()
		model.train()		
		loss = model.baseModel(X, musk, p1, p2, y)		
		loss_ = loss.item()	
		losses.append(loss_)
		if loss_!=loss_:
			return loss_, step
		optimizer.zero_grad()
		loss.backward()
		if args.clip!=0:
			nn.utils.clip_grad_norm(model.parameters(), args.clip)		
		optimizer.step()
		step = step + 1		
		print '[Epoch %d/%d ] | Iter %d/%d | Loss %f' % (epoch, args.epochs, step, all_step, loss_)
		if step%args.eval_per==0:
			save_checkpoint({
				'epoch': epoch,
				'step': step,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict()
			}, step)		
	return np.mean(np.array(losses)), step

def train_RE():
	data_loader = DataLoader(args.data_dir, args.max_bags, args.max_len, mode=args.mode, flip=args.flip)
	model = RE(args.dim, args.pos_dim, args.hdim, data_loader.n_word, data_loader.n_pos, data_loader.n_relation, args.reg_weight, reduce_method=args.reduce_method, position=args.position, encode_model=args.encoder, sigmoid=args.sigmoid, init1=args.init1, init2=args.init2, na_init1=args.na_init1, na_init2=args.na_init2)	
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	start_epoch = 0
	step = 0			
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']
		step = checkpoint['step']		
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	for epoch in range(start_epoch, args.epochs):
		loss, step = train_epoch_RE(model, data_loader, optimizer, epoch, step)
		if loss!=loss:
			print 'NaN loss'
			break
		print 'Epoch %d train over, average loss: %f' % (epoch, loss)
		with open(args.save_dir+'train_log.txt', 'ab') as f:
			f.write("[train] epoch %d step %d loss: %f\n" % (epoch, step, loss))

def train_EM():
	data_loader = DataLoader(args.data_dir, args.max_bags, args.max_len, mode=args.mode, flip=args.flip)	
	model = RE(args.dim, args.pos_dim, args.hdim, data_loader.n_word, data_loader.n_pos, data_loader.n_relation, args.reg_weight, reduce_method=args.reduce_method, position=args.position, encode_model=args.encoder, sigmoid=args.sigmoid, init1=args.init1, init2=args.init2, na_init1=args.na_init1, na_init2=args.na_init2)
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
	start_epoch = 0
	step = 0

	print 'executing e step...'	
	Q_y1_all = e_step(model, data_loader, init=True)	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		
	if args.resume != -1:
		checkpoint = resume(args.resume)
		start_epoch = checkpoint['epoch']+1
		step = checkpoint['step']		
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		Q_y1_all = np.load(args.save_dir+'Q_y.npy')

	print 'executing m step...'		
	for epoch in range(start_epoch, args.epochs):
		loss, step, Q_y1_all = em_step(model, data_loader, optimizer, epoch, step, Q_y1_all)
		print 'Epoch %d train over, average loss: %f' % (epoch, loss)		
		with open(args.save_dir+'train_log.txt', 'ab') as f:
			f.write("[train] epoch %d step %d loss: %f\n" % (epoch, step, loss))	

def test_model():
	data_loader = DataLoader(args.data_dir, args.max_bags, args.max_len, mode=args.mode, flip=args.flip)
	model = RE(args.dim, args.pos_dim, args.hdim, data_loader.n_word, data_loader.n_pos, data_loader.n_relation, args.reg_weight, reduce_method=args.reduce_method, position=args.position, encode_model=args.encoder, sigmoid=args.sigmoid, init1=args.init1, init2=args.init2, na_init1=args.na_init1, na_init2=args.na_init2)	
	if use_cuda:
		model = model.cuda()
	optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)	

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)		

	step = 500
	dataset = 'test'
	best_ap = 0.
	while step<=17500:
		checkpoint = resume(step)
		model.load_state_dict(checkpoint['state_dict'])	
		y_score,Y_test,ap, p,r,f1 = pred_RE(model, data_loader, dataset=dataset)
		if ap>=best_ap:
			best_ap = ap
			get_pre_recall(y_score,Y_test)
		print "step %d p: %f r: %f f1: %f Average Precision: %f" % (step, p,r,f1, ap)	
		with open(args.save_dir+'%s_log_pr.txt'%(dataset), 'ab') as f:
			f.write("step %d p: %f r: %f f1: %f Average Precision: %f\n" % (step, p,r,f1, ap))	
		step += 500	

	print 'best ap: %f'%best_ap
	with open(args.save_dir+'%s_log_pr.txt'%(dataset), 'ab') as f:
		f.write('best ap: %f\n'%best_ap)	

def main():
	print args
	if args.run_mode=='train':
		if args.em:
			train_EM()
		else:			
			train_RE()
		test_model()
	elif args.run_mode=='test':		
		test_model()

if __name__ == '__main__':
	main()	

