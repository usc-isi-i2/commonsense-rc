from mowgli.classes import Dataset, Entry
import mowgli.utils.general as utils
from mowgli.predictor.predictor import Predictor
from trian.trian_classes import SpacyTokenizer
from typing import List, Any
from trian.preprocess_utils import preprocess_dataset, preprocess_cskg
from trian.preprocess_utils import build_vocab
from trian.model import Model
from trian import config

import os
import time
import torch
import random
import numpy as np
from datetime import datetime

class Trian(Predictor):
	def preprocess(self, dataset:Dataset) -> Any:
		tok = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})
		preprocess_cskg('./data/cskg/edges_v004.csv', dataset, tok)
		print('done preprocessing cskg')
		for partition in ['train', 'dev', 'test']:
			part_data=getattr(dataset, partition)
			preprocess_dataset(part_data, partition, tok, partition=='test')
		return dataset

	def train(self, train_data:List, dev_data: List, graph: Any) -> Any:

		model_args=config.model_args
		if model_args.pretrained:
			assert all(os.path.exists(p) for p in model_args.pretrained.split(',')), 'Checkpoint %s does not exist.' % model_args.pretrained

		torch.manual_seed(model_args.seed)
		np.random.seed(model_args.seed)
		random.seed(model_args.seed)

		best_model=None

		build_vocab()
		if model_args.test_mode:
			# use validation data as training data
			train_data += dev_data
			dev_data = []
		model = Model(model_args)

		best_dev_acc = 0.0
		os.makedirs('./checkpoint', exist_ok=True)
		checkpoint_path = './checkpoint/%d-%s.mdl' % (args.seed, datetime.now().isoformat())
		print('Trained model will be saved to %s' % checkpoint_path)

		for i in range(model_args.epoch):
			print('Epoch %d...' % i)
			if i == 0:
				dev_acc = model.evaluate(dev_data)
				print('Dev accuracy: %f' % dev_acc)
			start_time = time.time()
			np.random.shuffle(train_data)
			cur_train_data = train_data

			model.train(cur_train_data)
			train_acc = model.evaluate(train_data[:2000], debug=False, eval_train=True)
			print('Train accuracy: %f' % train_acc)
			dev_acc = model.evaluate(dev_data, debug=True)
			print('Dev accuracy: %f' % dev_acc)

			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				os.system('mv ./data/output.log ./data/best-dev.log')
				model.save(checkpoint_path)
				best_model=copy.deepcopy(model)
			elif model_args.test_mode:
				model.save(checkpoint_path)
			print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

		print('Best dev accuracy: %f' % best_dev_acc)
		return best_model

	def predict(self, model: Any, entry: Entry) -> List:
		question=entry.question
		answers=entry.answers

		answer=random.randint(0,len(answers)-1)
		while answers[answer]=='':
			answer=random.randint(0,len(answers)-1)
		probs=['%.2f' % random.random() for i in range(len(answers))]
		return str(answer), probs
