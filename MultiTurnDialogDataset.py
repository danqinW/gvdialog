import pickle
import os

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.vocab import Vocab
from utils.pad import *

class MultiTurnDialogDataset(Dataset):

    def __init__(
        self, 
        conversations,
        tokenizer,
        vocabulary: Vocab, 
        max_conv_len=15,
        max_seq_len=10,
        append_eos=True, 
        build_vocab=True,
        save_cache=True,
        cache_path=None
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.vocabulary = vocabulary
        self.max_conv_len = max_conv_len
        self.max_seq_len = max_seq_len
        self.append_eos = append_eos
        self.build_vocab = build_vocab
        self.save_cache = save_cache
        self.cache_path = cache_path

        if cache_path and os.path.exists(cache_path):
            print('load cache...')
            with open(cache_path, 'rb') as f:
                self.context_ids, self.trg_ids, self.input_conv_length, \
                    self.input_sen_length, self.trg_sen_length = pickle.load(f)

        
        else:

            for i, c in tqdm(enumerate(conversations), total=len(conversations)):
                conversations[i] = [vocabulary.encode_line(
                    line,
                    tokenizer=tokenizer,
                    append_eos=append_eos,
                    add_if_not_exist=self.build_vocab
                ) for line in c]


            context_ids = []
            trg_ids = []
            for ind, conv in enumerate(conversations):
                # if not all(len(s) < max_seq_len for s in conv):
                #     continue
                if len(conv) < 2:
                    continue
                if len(conv) > max_conv_len:
                    conv = conv[:max_conv_len]
                
                sen_lengths = [min(len(s), max_seq_len) for s in conv]
                conv = [s[:max_seq_len-1]+[vocabulary.eos_id] if len(s) > max_seq_len else s for s in conv]
                
                for i in range(1, len(conv)):
                    context = conv[:i]
                    response = conv[i]
                    context_ids.append(context)
                    trg_ids.append(response)
            
            # drop duplicate
            dup = set()
            self.context_ids = []
            self.trg_ids = []
            self.input_conv_length = []
            self.input_sen_length = []
            self.trg_sen_length = []
            for _, (ctx, resp) in tqdm(enumerate(zip(context_ids, trg_ids)), total=len(context_ids)):
                ctx_str = ''.join([''.join(map(lambda w: chr(w), sen)) for sen in ctx])
                resp_str = ''.join(map(lambda w: chr(w), resp))
                value = hash(ctx_str + resp_str)
                if value in dup:
                    continue
                dup.add(value)
                self.context_ids.append(ctx)
                self.trg_ids.append(resp)
                self.input_conv_length.append(len(ctx))
                self.input_sen_length.append([len(c) for c in ctx])
                self.trg_sen_length.append(len(resp))

            # save cache
            if self.save_cache and self.cache_path:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump((
                        self.context_ids,
                        self.trg_ids, 
                        self.input_conv_length,
                        self.input_sen_length, 
                        self.trg_sen_length
                    ), f)

    
    def __len__(self):
        return len(self.context_ids)
    
    def __getitem__(self, index):
        context_ids = self.context_ids[index]
        trg_ids = [self.vocabulary.sos_id] + self.trg_ids[index]
        return (context_ids, self.input_conv_length[index], self.input_sen_length[index], 
                trg_ids, self.trg_sen_length[index])

    @staticmethod
    def collate(data: list):
        context_ids, conv_lens, sentence_lens, trg_ids, trg_lens = list(zip(*data))
        input_ids = [s for c in context_ids for s in c]

        input_sentence_lens = torch.tensor([l for s in sentence_lens for l in s]).long()
        input_conv_lens = torch.tensor(conv_lens).long()
        

        input_var = zero_padding(input_ids)
        input_var = torch.LongTensor(input_var)

        max_output_len = max(trg_lens)
        output_var = zero_padding(trg_ids)
        output_mask = binary_mask(output_var)
        output_var = torch.LongTensor(output_var)
        output_mask = torch.BoolTensor(output_mask)

        return input_var, input_conv_lens, input_sentence_lens, output_var, output_mask, max_output_len
