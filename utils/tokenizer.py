import re
import unicodedata
from .vocab import *


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def init_clean(string):
    string = unicodeToAscii(string)
    string = re.sub(r"\.{3}", " ... ", string)
    string = re.sub(r'</?[uib]>', ' ', string)
    string = re.sub(r"[^a-zA-Z\.\'\"\!\?\,\-]", ' ', string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = init_clean(string)
    string = re.sub(r'([^\.\s])\.', r'\1 .', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'\s", " \' ", string)
    string = re.sub(r"\s\'([^smvtrdl\s])", r" \' \1", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if _is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

class Tokenizer(object):

    def __init__(self, tokenize_type='gpt2') -> None:
        self.tokenize_type = tokenize_type
        if tokenize_type == 'spacy':
            import spacy
            nlp = spacy.load('en_core_web_sm')
            self.tokenizer = lambda x: [t.text for t in nlp(init_clean(x))]
            self.aggregator = lambda x: ' '.join(x)
        elif tokenize_type == 'gpt2':
            from transformers import GPT2Tokenizer
            self.gpt = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer = lambda x: self.gpt.tokenize(init_clean(x))
            self.aggregator = lambda tokens: self.gpt.convert_tokens_to_string(tokens)
        # elif tokenize_type == 'bert'
        elif tokenize_type == 'zh':
            self.tokenizer = lambda x: [ch for ch in x if ch.isalnum() or _is_chinese_char(ord(ch))]
            self.aggregator = lambda x: ''.join(x)
        else:
            self.tokenizer = lambda x: x.split()
            self.aggregator = lambda x: ' '.join(x)
        
    def tokenize(self, line):
        line = self.tokenizer(line)
        return line.split(' ')
    
    def aggregate(self, tokens):
        return self.aggregator(tokens)

