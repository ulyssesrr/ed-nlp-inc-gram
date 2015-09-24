import re
import functools

PATTERN = re.compile("(?P<word>\w+)|\<(?P<lemma>\w+)?\.(?P<pos>[\w\+]+)?(\:(?P<inflectional>\w+))?\>")

class Grammar:
    
    def __init__(self, tokenized_grammar):
        self.compiled_grammar = self.compile_grammar(tokenized_grammar)
    
    def compile_grammar(self, tokenized_grammar):
        compiled_grammar = []
        for gt in tokenized_grammar:
            compiled_grammar += [self.compile_grammar_token(gt)]
        return compiled_grammar
    
    
    def compile_grammar_token(self, gt):
        d = PATTERN.match(gt)
        if d == None:
            raise Exception("Invalid pattern: %s" % gt)
        d = d.groupdict()
        ret = []
        if d["word"] != None:
            f = functools.partial(self._match_word, expected_word=d['word'].lower())
            ret += [f]
        else:
            if d['pos'] != None:
                f = functools.partial(self._match_pos, expected_pos_tag=d['pos'].lower())
                ret += [f] 
            if d['lemma'] != None:
                f = functools.partial(self._match_lemma, test_lemma=d['lemma'])
                ret += [f]
            if d['inflectional'] != None:
                f = functools.partial(self._match_inflectional, test_inf=d['inflectional'])
                ret += [f]
        return ret
    
    def _match_pos(self, lexicon, word, pos_tag, expected_pos_tag):
        return pos_tag.lower() == expected_pos_tag
    
    def _match_word(self, lexicon, word, pos_tag, expected_word):
        return word.lower() == expected_word
    
    def _match_lemma(self, lexicon, word, pos_tag, test_lemma):
        lemmas = lexicon.get_lemmas(word)
        for lemma, info in lemmas:
            if lemma == test_lemma:
                return True
        return False
    
    def _match_inflectional(self, lexicon, word, pos_tag, test_inf):
        lemmas = lexicon.get_lemmas(word)
        for lemma, info in lemmas:
            d = PATTERN.match("<%s>" % info)
            if d == None:
                raise Exception("Unsupported pattern: %s" % info)
            d = d.groupdict()
            if test_inf in d['inflectional']:
                return True
        return False
    
    def match(self, lexicon, tagged_sent):
        len_grammar = len(self.compiled_grammar)
        len_tsent = len(tagged_sent)
        if len_grammar <= len_tsent:
            for i in range(len_tsent - len_grammar + 1):
                if self._match_tokens(lexicon, tagged_sent[i:i+len_grammar]):
                    return i
        return -1
    
    def _match_tokens(self, lexicon, tsent_tokens):
        for cgt, tst in zip(self.compiled_grammar, tsent_tokens):
            if not Grammar._match_token(lexicon, cgt, tst):
                return False
        return True
    
    @staticmethod
    def _match_token(lexicon, cgt, tst):
        for f in cgt:
            #print(f, tst, f(lexicon, *tst))
            if not f(lexicon, *tst):
                return False
        return True