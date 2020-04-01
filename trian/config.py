from attrdict import AttrDict

def get_model_args(dataset='se2018t11', kg='conceptnet'):

	model_args={
			'gpu': '0',
			'epoch': 50,
			'optimizer': 'adamax',
			'use_cuda': True,
			'grad_clipping': 10.0,
			'lr': 2e-3,
			'batch_size': 32,
			'embedding_file': './data/glove.840B.300d.txt',
			'hidden_size': 96,
			'doc_layers': 1,
			'rnn_type': 'lstm',
			'dropout_rnn_output': 0.4,
			'rnn_padding': True,
			'dropout_emb': 0.4,
			'pretrained': '',
			'finetune_topk': 10,
			'pos_emb_dim': 12,
			'ner_emb_dim': 8,
			'rel_emb_dim': 10,
			'seed': 1234,
			'test_mode': False,
			'checkpoint_dir': './checkpoint',
			'last_log': './output/%s-%s/output.log' % (kg,dataset),
			'best_log': './output/%s-%s/best-dev.log' % (kg,dataset),
			'save_loc': './output/%s-%s/model.mdl' % (kg, dataset)
			}
	return AttrDict(model_args)

def get_pp_args(dataset='se2018t11', kg='conceptnet'):
	preprocessing_args={
			'kg_filtered': './output/%s-%s/%s.filter' % (kg, dataset, kg),
			'kg_edges': './data/%s/edges_v004.csv' % kg,
			'partitions': ['train', 'dev', 'test', 'trial'],
			'vocab_file': './output/%s-%s/vocab' % (kg,dataset),
            'rel_vocab_file': './output/%s-%s/rel_vocab' % (kg,dataset),
            'pos_vocab_file': './output/%s-%s/pos_vocab' % (kg,dataset),
            'ner_vocab_file': './output/%s-%s/ner_vocab' % (kg,dataset),
            'race_vocab_file': './output/%s-%s/race_vocab' % (kg,dataset),
			'processed_file': './output/' + kg + '-' + dataset + '/%s-processed.json'
			}

	return AttrDict(preprocessing_args)

