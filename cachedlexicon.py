from functools import lru_cache

class CachedLexicon():

	def __init__(self, lexicon):
		self.lexicon = lexicon
	
	@lru_cache(maxsize=1024*1024)
	def get_lemmas(self, word):
		return self.lexicon.get_lemmas(word)