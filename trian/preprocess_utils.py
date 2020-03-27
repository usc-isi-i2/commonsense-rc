import os
import sys
import json
import math
import glob
import wikiwords
from collections import Counter

from trian.trian_classes import SpacyTokenizer 
from trian.utils import is_stopword, is_punc
from trian import utils
####################### PREPROCESSING DATASET ###############

digits2w = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
def replace_digits(words):
    global digits2w
    return [digits2w[w] if w in digits2w else w for w in words]

def tokenize(text, tok):
    """Call the global process tokenizer on the input text."""
    tokens = tok.tokenize(text)
    output = {
        'words': replace_digits(tokens.words()),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output

def compute_features(d_dict, q_dict, c_dict):
    # in_q, in_c, lemma_in_q, lemma_in_c, tf
    q_words_set = set([w.lower() for w in q_dict['words']])
    in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['words']]
    c_words_set = set([w.lower() for w in c_dict['words']])
    in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['words']]

    q_words_set = set([w.lower() for w in q_dict['lemma']])
    lemma_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['lemma']]
    c_words_set = set([w.lower() for w in c_dict['lemma']])
    lemma_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in d_dict['lemma']]

    tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in d_dict['words']]
    tf = [float('%.2f' % v) for v in tf]
    d_words = Counter(filter(lambda w: not is_stopword(w) and not is_punc(w), d_dict['words']))
    from trian.kg import my_kg
    p_q_relation = my_kg.p_q_relation(d_dict['words'], q_dict['words'])
    p_c_relation = my_kg.p_q_relation(d_dict['words'], c_dict['words'])
    assert len(in_q) == len(in_c) and len(lemma_in_q) == len(in_q) and len(lemma_in_c) == len(in_q) and len(tf) == len(in_q)
    assert len(tf) == len(p_q_relation) and len(tf) == len(p_c_relation)
    return {
        'in_q': in_q,
        'in_c': in_c,
        'lemma_in_q': lemma_in_q,
        'lemma_in_c': lemma_in_c,
        'tf': tf,
        'p_q_relation': p_q_relation,
        'p_c_relation': p_c_relation
    }

def get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label):
    return {
            'id': d_id + '_' + q_id + '_' + c_id,
            'd_words': ' '.join(d_dict['words']),
            'd_pos': d_dict['pos'],
            'd_ner': d_dict['ner'],
            'q_words': ' '.join(q_dict['words']),
            'q_pos': q_dict['pos'],
            'c_words': ' '.join(c_dict['words']),
            'label': label
        }

def preprocess_dataset(data, partition, tok, is_test_set=False):
	writer = open('./data/%s-processed.json' % partition, 'w', encoding='utf-8')
	ex_cnt = 0
	for obj in data:
		q_id = getattr(obj, 'id')
		d_id = q_id
		context, question=obj.question
		d_dict = tokenize(context, tok)
		q_dict = tokenize(question, tok)
		answers=getattr(obj, 'answers')
		for i, ans in enumerate(answers):
			c_dict = tokenize(ans, tok)
			correct_answer=getattr(obj, 'correct_answer')
			label = answers[int(correct_answer)] if not is_test_set else -1
			c_id = correct_answer
			example = get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label)
			example.update(compute_features(d_dict, q_dict, c_dict))
			writer.write(json.dumps(example))
			writer.write('\n')

			ex_cnt += 1
	print('Found %d examples in %s...' % (ex_cnt, path))
	writer.close()

################ Preprocessing CSKG ###################

def build_vocab(dataset, tok):
	word_cnt = Counter()
	for partition in ['train', 'dev', 'test']:
		part_data=getattr(dataset, partition)
		for obj in part_data:
			context, question=obj.question
			d_dict = tokenize(context, tok)
			word_cnt += Counter(d_dict['words'])
			q_dict = tokenize(question, tok)
			word_cnt += Counter(q_dict['words'])
			for ans in getattr(obj, 'answers'):
				c_dict = tokenize(ans, tok)
				word_cnt += Counter(c_dict['words'])
	for key, val in word_cnt.most_common():
		utils.vocab.add(key)
	print('Vocabulary size: %d' % len(utils.vocab))
	writer = open('./data/vocab', 'w', encoding='utf-8')
	writer.write('\n'.join(utils.vocab.tokens()))
	writer.close()

def preprocess_cskg(path, dataset, tok):
    build_vocab(dataset, tok)
    writer = open('./data/cskg.filter', 'w', encoding='utf-8')
    def _get_w(arg, labels):
        if arg in labels.keys():
            return labels[arg]
        else: return ''

    nodes_path=path.replace('edges', 'nodes')
    node2label={}
    with open(nodes_path, 'r') as f:
        next(f)
        for line in f:
            fs=line.split('\t')
            node2label[fs[0]]=fs[1]
    print('node index ready')
    datasources=[]
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            fs = line.split('\t')
            if fs[3]=='weight': continue # header column

            arg1, relation, arg2 = fs[0], utils.get_uri_meat(fs[1]), fs[2]
            w1 = _get_w(arg1, node2label)
            if not all(w in utils.vocab for w in w1.split(' ')):
                continue
            w2 = _get_w(arg2, node2label)
            if not all(w in utils.vocab for w in w2.split(' ')):
                continue
            if float(fs[3])<1.0 or w1==w2: # weight<1.0 or same words -> skip
                continue
            datasources.append(fs[4])
            writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()
    print(Counter(datasources))
