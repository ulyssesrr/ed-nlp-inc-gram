#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
from pydela.lexicon import Lexicon
import itertools
import re

#lexicon = Lexicon("/home/ulysses/Applications/Unitex3.1beta/Portuguese (Brazil)/Dela/")
lexicon = Lexicon("/home/ulysses/Apps/Unitex3.1beta/Portuguese (Brazil)/Dela/")

def get_grammar_candidates(sentence):
	p_pos = re.compile("(^\w+)")
	
	p_inf_lower = re.compile("\:[A-Z0-9]*([a-z]+)")
	p_inf_upper = re.compile("\:([A-Z0-9]+)")
	p_inf_full = re.compile("\:(\w+)")
	
	pos_rules = [
		lambda t : ".%s" % p_pos.findall(t[1])[0],
		lambda t : "%s." % t[0],
		lambda t : "%s.%s" % (t[0], p_pos.findall(t[1])[0]),
	]
	
	inflection_rules = [
		lambda t : [""], # no inflection
		lambda t : p_inf_lower.findall(t[1]),
		lambda t : p_inf_upper.findall(t[1]),
		lambda t : p_inf_full.findall(t[1])
	]
	
	
	
	tokenizer = re.compile('\w+')
	
	lemmas = list(lexicon.get_lemmas("o"))[0]
	print(lemmas)
		
	
	
	
	#rl_simple = lambda p : p[1].split(":")[0].split("+")[0]
	#rl_inflectional_full = lambda p : p_med.sub("",p[1])
	#rl_inflectional_single = lambda p : p_med.sub("",p[1])
	#rules = [
		#rl_simple,
		#rl_inflectional_full,
		#lambda p : "%s.%s" % (p[0], rl_simple(p)),
		#lambda p : "%s.%s" % (p[0], rl_inflectional_full(p))
	#]
	
	sent_words = tokenizer.findall(sentence)
	for w in sent_words:
		lemmas = lexicon.get_lemmas(w)
		grammar_sufixes = []
		grammar_prefixes = []
		for lemma in lemmas:
			for ir in inflection_rules:
				grammar_sufixes += ir(lemma)
			
			grammar_sufixes = list(set(grammar_sufixes))
			grammar_prefixes += [pr(lemma) for pr in pos_rules]
		
		print(grammar_prefixes)
		print(grammar_sufixes)
	
	
	candidates_simple = set(itertools.product(*pos_simple))
	candidates_med = set(itertools.product(*pos_med))
	candidates_full = set(itertools.product(*pos_full))
		#print("ITERTOOLS")
		#print(candidates_simple)
	return candidates_simple, candidates_med, candidates_full

print(get_grammar_candidates("O gato"))