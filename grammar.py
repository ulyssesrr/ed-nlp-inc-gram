import re

PATTERN = re.compile("(?P<word>\w+)|\<(?P<lemma>\w+)?\.(?P<pos>\w+)?(\:(?P<inflectional>\w+))?\>")

class Grammar:
	
	def __init__(self, tokenized_grammar):
		self.compiled_grammar = Grammar.compile_grammar(tokenized_grammar)
		
	@staticmethod
	def compile_grammar(tokenized_grammar):
		compiled_grammar = []
		for gt in tokenized_grammar:
			compiled_grammar += [Grammar.compile_grammar_token(gt)]
		return compiled_grammar
	
	@staticmethod
	def compile_grammar_token(gt):
		d = PATTERN.match(gt)
		if d == None:
			raise Exception("Invalid pattern: %s" % gt)
		d = d.groupdict()
		ret = []
		if d["word"] != None:
			ret += [lambda x : x[0].lower() == d['word'].lower()]
		else:
			if d['pos'] != None:
				ret += [lambda x : x[1].lower() == d['pos'].lower()]
			if d['lemma'] != None:
				raise Exception("Not implemented yet")
			if d['inflectional'] != None:
				raise Exception("Not implemented yet")
				#ret += [lambda x : x[1].lower() == d['word'].lower()]
		return ret
	
	def match(self, tagged_sent):
		len_grammar = len(self.compiled_grammar)
		len_tsent = len(tagged_sent)
		if len_grammar <= len_tsent:
			for i in range(len_tsent - len_grammar + 1):
				if self._match_tokens(tagged_sent[i:i+len_grammar]):
					return i
		return -1
	
	def _match_tokens(self, tsent_tokens):
		for cgt, tst in zip(self.compiled_grammar, tsent_tokens):
			if not Grammar._match_token(cgt, tst):
				return False
		return True
	
	@staticmethod
	def _match_token(cgt, tst):
		for f in cgt:
			print(f, tst, f(tst))
			if not f(tst):
				return False
		return True