class Grammar:
	
	def __init__(self, tokenized_grammar):
		self.grammar = tokenized_grammar
	
	def match(self, tagged_sent):
		len_grammar = len(self.grammar)
		len_tsent = len(tagged_sent)
		if len_grammar <= len_tsent:
			for i in range(len_tsent - len_grammar):
				if _match_tokens(tagged_sent[i:i+len_grammar]):
					return i
		return -1
	
	def _match_tokens(self, tsent_tokens):
		for gt, tst in zip(self.grammar, tsent_tokens):
			if not _match_token(gt, tst):
				return False
	
	def _match_token(gt, tst):
		pos_tag = 