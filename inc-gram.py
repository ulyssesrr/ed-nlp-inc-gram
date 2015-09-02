#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
from __future__ import print_function
import sys
import gc
import resource
import re
import logging
import pickle
import math
import time
import itertools
from datetime import timedelta

from multiprocessing import Pool
import functools

from optparse import OptionParser

import numpy as np

from scipy import sparse as sp

import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from pydela.lexicon import Lexicon	

import nltk
from nltk.cluster import KMeansClusterer
from nltk.corpus import mac_morpho

from sklearn.cluster import KMeans

parser = OptionParser(usage="%prog [options] <datasetdir>")
parser.add_option("-e", "--encoding", dest="encoding", default="latin_1", help="Dataset encoding")
parser.add_option("-r", "--initial-ranking", dest="ranking_method", default="cosine_similarity", help="Initial ranking method (cosine_similarity, accuracy) Default: cosine_similarity")

if sys.stdout.encoding == None:
	print("Fixing stdout encoding...")
	import codecs
	import locale
	# Wrap sys.stdout into a StreamWriter to allow writing unicode.
	sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout)

(options, args) = parser.parse_args()

if len(args) == 0:
	parser.print_help()
	sys.exit()

class ClusteringSelectStrategy:
	
	def select_grammar(self, vectors, candidates_simple, sorted_grams_idx, gram_rank):
		#clusterer = KMeansClusterer(2, nltk.cluster.util.euclidean_distance)#, initial_means=means) 
		#clusters = clusterer.cluster(vectors.todense(), True, trace=True) 
		#print('Clusters:', clusters )
		#print('Means:', clusterer.means())
		log.info("Clustering...")
		n_clusters = 2
		kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
		kmeans.fit(vectors)
		order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
		print(order_centroids)
		for i in range(2):
			print("Cluster %d:" % i, end='')
			for ind in order_centroids[i, :2]:
				print(candidates_simple[ind])
				
		log.info("Getting centers...")
		for label in range(n_clusters):
			label_idxs = np.where(kmeans.labels_ == label)
			c_idx = euclidean_distances(vectors[label_idxs], kmeans.cluster_centers_[i]).argmin(axis=0)[0]
			print(np.array(candidates_simple)[label_idxs])
			nearest = vectors[label_idxs][c_idx]
			print("Nearest %d: " % (label), nearest.todense(), np.array(candidates_simple)[label_idxs][c_idx])
		
class ClusteringSelectStrategy2:
	
	def select_grammar(self, vectors, candidates, sorted_grams_idx, gram_rank):
		#clusterer = KMeansClusterer(2, nltk.cluster.util.euclidean_distance)#, initial_means=means) 
		#clusters = clusterer.cluster(vectors.todense(), True, trace=True) 
		#print('Clusters:', clusters )
		#print('Means:', clusterer.means())
		log.info("Clustering...")
		n_clusters = math.floor(len(candidates)/2)
		kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
		kmeans.fit(gram_rank)
		order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
		#print(order_centroids)
		#for i in range(n_clusters):
			#print("Cluster %d:" % i, end='')
			#for ind in order_centroids[i, :2]:
				#print(candidates_simple[ind])
				
		log.info("Getting centers...")
		for label in range(n_clusters):
			label_idxs = np.where(kmeans.labels_ == label)
			#print("CENTER", kmeans.cluster_centers_[label])
			if label_idxs[0].shape[0] > 0:
				#print(label_idxs, )
				#print(gram_rank[label_idxs])
				c_idx = euclidean_distances(gram_rank[label_idxs], kmeans.cluster_centers_[label]).argmin(axis=0)[0]
				print(c_idx)
				nearest = vectors[label_idxs][c_idx]
				print("Nearest %d: " % (label), nearest.todense(), np.array(candidates_simple)[label_idxs][c_idx])
				abs_idx = np.arange(len(gram_rank))[label_idxs][c_idx]
				print("ABS Nearest %d: " % (label), nearest.todense(), candidates_simple[abs_idx])
		print("END")



class ElapsedFormatter():
	
	def __init__(self):
		self.start_time = time.time()
	
	def format(self, record):
		elapsed_seconds = record.created - self.start_time
		#using timedelta here for convenient default formatting
		elapsed = timedelta(seconds = elapsed_seconds)
		return "[%s][RAM: %.2f MB] %s" % (str(elapsed)[:-3], (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), record.getMessage())

#add custom formatter to root logger for simple demonstration
handler = logging.StreamHandler()
handler.setFormatter(ElapsedFormatter())
logging.getLogger().addHandler(handler)

log = logging.getLogger('main')
log.setLevel(logging.DEBUG)

with open("input.txt") as f:
	text = f.readlines()
	inputs = [i.replace("\n", "").lower().split(";") for i in text]
	log.info(inputs)

lexicon = Lexicon("/home/ulysses/Applications/Unitex3.1beta/Portuguese (Brazil)/Dela/")

def get_candidates(sentences):
	candidates_simple = set()
	candidates_med = set()
	candidates_full = set()
	tokenizer = re.compile('\w+')
	for s in sentences:
		sent_words = tokenizer.findall(s)
		pos_full = []
		pos_med = []
		pos_simple = []
		for w in sent_words:
			lemmas = lexicon.get_lemmas(w)
			pos_full += [set([p[1] for p in lemmas])]
			pos_med += [set([p[1].split(":")[0] for p in lemmas])]
			pos_simple += [set([p[1].split(":")[0].split("+")[0] for p in lemmas])]
			#print(w, lemmas)
			#print(pos_med)
			#print(pos_simple)
		
		if len(candidates_simple) == 0:
			#print("TESTE",pos_simple)
			candidates_simple = set(itertools.product(*pos_simple))
			candidates_med = set(itertools.product(*pos_med))
			candidates_full = set(itertools.product(*pos_full))
		else:
			candidates_simple = candidates_simple.intersection(set(itertools.product(*pos_simple)))
			candidates_med = candidates_med.intersection(set(itertools.product(*pos_med)))
			candidates_full = candidates_full.intersection(set(itertools.product(*pos_full)))
		#print("ITERTOOLS")
		#print(candidates_simple)
	return candidates_simple, candidates_med, candidates_full

sentences = [s[1] for s in inputs]

try:
	tagger2 = pickle.load(open( "tagger.pickle", "rb" ))
	log.info("Loaded pre-trained tagger.")
except:
	log.info("Loading Mac-Morpho Tagged Sents...")
	tsents = list(mac_morpho.tagged_sents())


	def simplify_tag(t):
		if "+" in t:
			t = t[t.index("+")+1:]
		
		if t == "ART":
			return "DET"
		
		return t

	log.info("Simplifyng POS Tags...")
	tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]

	train = tsents
	test = tsents[:300]
	log.info("Training POS Taggers...")
	tagger0 = nltk.DefaultTagger('N')
	tagger1 = nltk.UnigramTagger(train, backoff=tagger0)
	tagger2 = nltk.BigramTagger(train, backoff=tagger1)
	pickle.dump(tagger2, open( "tagger.pickle", "wb" ) )

#log.info("Evaluate tagger")
#print(tagger2.evaluate(test))

#log.info("TAGSET")
#tags = [simplify_tag(tag) for (word,tag) in mac_morpho.tagged_words()]
#fd = nltk.FreqDist(tags)
#print(fd.keys())

dataset_folder = args[0]

log.info("Loading dataset %s (encoding=%s)..." % (dataset_folder, options.encoding))
dataset = sklearn.datasets.load_files(dataset_folder, encoding=options.encoding, decode_error='strict')
log.info("[OK]")

log.info("Preprocessing Dataset: Cleaning HTML tags")
compiled_re = re.compile("<[^<]+?>")
dataset.data = [compiled_re.sub('', row) for row in dataset.data]

log.info("Preprocessing Dataset: to lowercase...")
dataset.data = [row.lower() for row in dataset.data[:1000]]

log.info("Tokenizing sentences in text...")
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
with Pool() as p:
	dataset_sentences = list(itertools.chain.from_iterable(p.map(sent_tokenizer.tokenize, dataset.data)))
print(dataset_sentences[0])
log.info("[%d]" % (len(dataset_sentences)))

log.info("Tokenizing words in sentences...")
tokenizer = re.compile('\w+')
with Pool() as p:
	tokenized_sents = list(p.map(tokenizer.findall, dataset_sentences))
	log.info("[OK]")
	log.info("Tagging dataset...")
	tagged_dataset = p.map(tagger2.tag, tokenized_sents)
	log.info("Done.")
print(tagged_dataset[0])

def find_sentence_match(tagged_sent, grammar):
	tags = [tag for word, tag in tagged_sent]
	if " ".join(grammar) in " ".join(tags):
		return tagged_sent
	return None

def find_dataset_match(tagged_dataset, grammar):
	with Pool() as p:
		f = functools.partial(find_sentence_match, grammar=grammar)
		return list(filter(lambda r : r != None, p.map(f, tagged_dataset)))

tokenizer = re.compile('\w+')
for input_id, s in enumerate(sentences):
	log.info("Sentence: %s" % (s))
	candidates_simple, candidates_med, candidates_full = get_candidates([s]) 
	candidates_simple, candidates_med, candidates_full = list(candidates_simple), list(candidates_med), list(candidates_simple)
	samples = []
	for i, cand_simple in enumerate(candidates_simple):
		sample = find_dataset_match(tagged_dataset, cand_simple)
		print(i, cand_simple, len(sample))
		if len(sample) == 0:
			del candidates_simple[i]
			del candidates_med[i]
			del candidates_full[i]
		else:
			samples += [sample]
			print(sample[0])
	
	candidates_simple, candidates_med, candidates_full = np.array(candidates_simple), np.array(candidates_med), np.array(candidates_simple)
	#print(candidates_simple)
	tagged_sent = [tagger2.tag([w])[0][1] for w in tokenizer.findall(s)]
	#print(s, tagged_sent)
	tagged_sent = np.array(tagged_sent)
	gram_acc = candidates_simple == tagged_sent
	#print(gram_acc, candidates_simple, tagged_sent)
	gram_acc = gram_acc.astype(np.float64).sum(axis=1) / gram_acc.shape[1]
	#print(gram_acc)
	
	log.info("Vectorizing...")
	count_vect = CountVectorizer(dtype=np.float64, token_pattern='\w+')
	X = [" ".join(tokens) for tokens in candidates_simple]
	#print(X)
	X_vect = count_vect.fit_transform(X)
	#print(X_vect.todense())
	tagged_sent_vect = count_vect.transform([" ".join(tagged_sent)])[0]
	#print(tagged_sent_vect)
	#print(X[0])
	#print(" ".join(tagged_sent))
	#print(tagged_sent_vect.todense())
	log.info("(%d, %d)" % (X_vect.shape[0],X_vect.shape[1]))
	
	gram_sim = cosine_similarity(X_vect, tagged_sent_vect)
	#print(gram_sim)
	
	if options.ranking_method == "cosine_similarity":
		log.info("Using cosine_similarity ranking...")
		gram_rank = gram_sim
	elif options.ranking_method == "accuracy":
		log.info("Using accuracy ranking...")
		gram_rank = gram_acc
	else:
		log.warning("Unknown ranking method %s ignored, using cosine_similarity" % (options.ranking_method))
		gram_rank = gram_sim
	
	top_idx = np.argmax(gram_rank)
	top_gram = candidates_simple[top_idx]
	#print(top_gram)
	sorted_grams_idx = np.argsort(-gram_rank, axis=0)
	
	selection_strategy = ClusteringSelectStrategy2()
	selection_strategy.select_grammar(X_vect, candidates_simple, sorted_grams_idx, gram_rank)
	
	log.info("%s: Writing results..." % (inputs[input_id]))
	with open("gram-%d.txt" % (input_id), 'w') as f:
		f.write("%s\n" % (inputs[input_id]))
		f.write("Meta: %s\n" % (" ".join(tagged_sent)))
		
		#print("sorted_grams_idx",sorted_grams_idx)
		for i, gram_idx in enumerate(sorted_grams_idx):
			gram = candidates_simple[gram_idx][0]
			#print(gram_idx, gram)
			f.write("%d: %s - %.03f\n" % (i, " ".join(gram), gram_rank[gram_idx]))
log.info("Finished")
	
