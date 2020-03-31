from mowgli.classes import Dataset, Entry
import mowgli.utils.general as utils
from mowgli.predictor.predictor import Predictor
from trian.trian_classes import SpacyTokenizer
from typing import List, Any
from trian.preprocess_utils import preprocess_dataset, preprocess_cskg
from trian.preprocess_utils import build_vocab
from trian.utils import load_vocab, load_data
from trian.model import Model
from trian.config import get_pp_args, get_model_args

import copy
import os
import time
import torch
import random
import numpy as np
from datetime import datetime

class Trian(Predictor):
	def preprocess(self, dataset:Dataset, kg='conceptnet') -> Any:
		pp_args=get_pp_args(dataset.name)
		tok = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})
		# Build vocabulary
		build_vocab(dataset, pp_args, tok)

		# Preprocess KG
		preprocess_cskg(pp_args)
		print('done preprocessing cskg')
		
		# Preprocess datasets
		preprocess_dataset(dataset, pp_args, tok)
		
		# Done
		return dataset

	def train(self, train_data:List, dev_data: List, graph: Any) -> Any:

		model_args=get_model_args(dataname)
		pp_args=get_pp_args(dataset.name)
		print('Model arguments:', model_args)
		if model_args.pretrained:
			assert all(os.path.exists(p) for p in model_args.pretrained.split(',')), 'Checkpoint %s does not exist.' % model_args.pretrained

		train_data = load_data(pp_args.processed_file % 'train')
		train_data += load_data(pp_args.processed_file % 'trial')
		dev_data = load_data(pp_args.processed_file % 'dev') 

		load_vocab(pp_args, train_data+dev_data)

		torch.manual_seed(model_args.seed)
		np.random.seed(model_args.seed)
		random.seed(model_args.seed)

		best_model=None

		if model_args.test_mode:
			# use validation data as training data
			train_data += dev_data
			dev_data = []
		model = Model(model_args)

		best_dev_acc = 0.0
		os.makedirs(model_args.checkpoint_dir, exist_ok=True)
		checkpoint_path = '%s/%d-%s.mdl' % (model_args.checkpoint_dir, model_args.seed, datetime.now().isoformat())
		print('Trained model will be saved to %s' % checkpoint_path)
		for i in range(model_args.epoch):
			print('Epoch %d...' % i)
			if i == 0:
				print('Dev data size', len(dev_data))
				dev_acc, dev_preds, dev_probs = model.evaluate(dev_data)
				print('Dev accuracy: %f' % dev_acc)
			start_time = time.time()
			np.random.shuffle(train_data)
			cur_train_data = train_data
		
			model.train(cur_train_data)
			train2000=train_data[:2000]
			train_acc, *rest = model.evaluate(train2000, debug=False, eval_train=True)
			print('Train accuracy: %f' % train_acc)
			dev_acc, dev_preds, dev_probs = model.evaluate(dev_data, debug=True)
			print('Dev accuracy: %f' % dev_acc)

			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				os.system('mv %s %s ' % (model_args.last_log, model_args.best_log))
				model.save(checkpoint_path)
				#best_model = Model(model_args)
				#best_model.network.load_state_dict(copy.deepcopy(model.network.state_dict()))
			elif model_args.test_mode:
				model.save(checkpoint_path)
			print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

		print('Best dev accuracy: %f' % best_dev_acc)

		return model

	def predict(self, model: Any, dataset: Dataset, partition: str) -> List:

		pp_args=AttrDict(pargs)
		dev_data = load_data(pp_args.processed_file % partition)

		dev_acc, dev_preds, dev_probs = model.evaluate(dev_data)
		print('Predict fn: Dev accuracy: %f' % dev_acc)

		#print(dev_preds)
		#print(dev_probs)
		return dev_preds, dev_probs
