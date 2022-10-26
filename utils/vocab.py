from tqdm import tqdm

PAD_TOKEN = '[PAD]'
SOS_TOKEN = '[SOS]'
EOS_TOKEN = '[EOS]'
UNK_TOKEN = '[UNK]'

class Vocab(object):
    
    def __init__(self, 
        pad_token=PAD_TOKEN, 
        sos_token=SOS_TOKEN, 
        eos_token=EOS_TOKEN, 
        unk_token=UNK_TOKEN,
    ):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.symbols = []
        self.counts = []
        self.word2index = {}
        self.pad_id = self.add_word(pad_token, 0)
        self.sos_id = self.add_word(sos_token, 0)
        self.eos_id = self.add_word(eos_token, 0)
        self.unk_id = self.add_word(unk_token, 0)
        
        self.nspeial = len(self.symbols)
        self.trimmed = False
       
    def add_word(self, w, n):
        if w not in self.word2index:
            self.word2index[w] = len(self.symbols)
            self.symbols.append(w)
            self.counts.append(n)
        else:
            ind = self.word2index[w]
            self.counts[ind] += n
        return self.word2index[w]
    
    def index(self, w):
        if w in self.word2index:
            return self.word2index[w]
        return self.unk_id

    def __getitem__(self, ind):
        if ind < len(self.symbols):
            return self.symbols[ind]
        return self.unk_token

    def __contains__(self, sym):
        return sym in self.word2index
    
    def __len__(self):
        return len(self.symbols)

    def add_sentence(self, s: list):
        for token in s:
            self.add_word(token, 1)
    
    def encode_line(self, line, tokenizer, append_eos=True, add_if_not_exist=True):
        words = tokenizer(line)
        ids = []
        for w in words:
            if add_if_not_exist:
                idx = self.add_word(w, 1)
            else:
                idx = self.index(w)
            ids.append(idx)
        if append_eos:
            ids.append(self.eos_id)
        return ids

    def decode_line(
        self, 
        token_ids, 
        aggregator,
        remove_eos=True,
        to_string=False,
    ):
        new_token_ids = []
        if remove_eos:
            for t in token_ids:
                if t == self.eos_id: break
                if t != self.pad_id:
                    new_token_ids.append(t)
        token_ids = new_token_ids
        res = [self.symbols[idx] for idx in token_ids]
        if to_string:
            res = aggregator(res)
        return res

    def size(self):
        return len(self.symbols)

    def trim(self, min_count):
        ''' set min_count threshold'''
        if self.trimmed: return
        self.trimmed = True
        
        new_symbols = []
        new_counts = []
        for k, c in zip(self.symbols[self.nspeial:], self.counts[self.nspeial:]):
            if c > min_count:
                new_symbols.append(k)
                new_counts.append(c)
        
        self.symbols = self.symbols[:self.nspeial] + new_symbols
        self.counts = self.counts[:self.nspeial] + new_counts
        self.word2index = {sym: i for i, sym in enumerate(new_symbols)}
    
    def keep_freq_n(self, n):
        '''keep top n frequent words except special tokens'''
        if self.trimmed: return
        self.trimmed = True

        first_n = n - self.nspeial
        new_symbols_and_counts = sorted(
            zip(self.symbols[self.nspeial:], self.counts[self.nspeial:]), 
            key=lambda x: x[1], 
            reverse=True
        )[: first_n]
        new_symbols, new_counts = list(zip(*new_symbols_and_counts))
        self.symbols = self.symbols[:self.nspeial] + list(new_symbols)
        self.counts = self.counts[:self.nspeial] + list(new_counts)

        self.word2index = {sym : i for i, sym in enumerate(self.symbols)}

    def save_vocab(self, output_path):
        with open(output_path, 'w', encoding='utf8') as f:
            for i in range(self.size()):
                f.write(self.symbols[i] + '\t' + str(self.counts[i]) + '\n')
    
    @classmethod
    def load_vocab(cls, vocab_path):
        voc = cls()
        with open(vocab_path, 'r', encoding='utf8') as f:
            for line in f:
                word, count = line.strip().split('\t')
                # word, count = line.strip(), 1
                voc.add_word(word, int(count))
        return voc
    
    @classmethod
    def add_file_to_vocab(cls, filename, tokenizer, separator=' __eou__ '):
        vocab = cls()
        with open(filename, 'r', encoding='utf8') as f:
            convs = [line.strip().split(separator) for line in f]
            for conv in tqdm(convs, total=len(convs)):
                for sen in conv:
                    vocab.add_sentence(tokenizer(sen))
        return vocab


