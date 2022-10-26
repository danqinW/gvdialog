from collections import defaultdict
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch
from utils.vocab import Vocab
from utils.tokenizer import Tokenizer
from MultiTurnDialogDataset import MultiTurnDialogDataset
from model import build_models

class BaseTask:
    def __init__(self, config, vocabulary: Vocab, tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.voc = vocabulary
        self.tokenizer = tokenizer
        self.dataloader = {}
        self.epoch_log_out = defaultdict(list)
        self.log = defaultdict(list)

    @classmethod
    def setup_task(cls, config):
        tokenizer = Tokenizer(config.tokenizer_type)
        if not config.vocab_path:
            if config.corpus_path == '' or not os.path.exists(os.path.join(config.corpus_path, 'train.txt')):
                raise Exception("can't build vocabulary")
            vocabulary = cls.build_vocab(config, tokenizer)
            if not os.path.exists(config.cachedir):
                os.makedirs(config.cachedir)
            vocabulary.save_vocab(os.path.join(config.cachedir,'vocab.txt'))
        else:
            print('load vocabulary...')
            vocabulary = Vocab.load_vocab(os.path.join(config.vocab_path, 'vocab.txt'))
        config.voc = vocabulary
        return cls(config, vocabulary, tokenizer)
    
    @classmethod
    def build_vocab(cls, config, tokenizer):
        train_file = os.path.join(config.corpus_path, 'train.txt')
        vocab = Vocab.add_file_to_vocab(train_file, tokenizer=tokenizer.tokenizer, separator=config.separator)
        if config.vocab_size:
            vocab.keep_freq_n(config.vocab_size)
        elif config.min_count:
            vocab.trim(config.min_count)
        config.vocab_size = len(vocab)
        return vocab

    def build_model(self, config):
        model = build_models(config.model, config)
        return model
    
    def build_criterion(self, config):
        from criterion.cross_entropy import CrossEntropy
        loss = CrossEntropy(config.label_smoothing, self.voc.pad_id)
        return loss

    def load_dataset(self, split, separator=' __eou__ '):
        conversations = []
        cache_path = os.path.join(self.config.cachedir, f'{split}.pkl')
        if not os.path.exists(cache_path):
            with open(os.path.join(self.config.corpus_path, f'{split}.txt'), 'r', encoding='utf8')as f:
                conversations = [line.split(separator) for line in f]
        dataset = MultiTurnDialogDataset(
            conversations, 
            self.tokenizer.tokenizer,
            self.voc,
            self.config.max_conv_len,
            self.config.max_seq_len,
            build_vocab=False,
            save_cache=self.config.save_cache,
            cache_path=cache_path
        )
        self.dataloader[split] = DataLoader(
            dataset,
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=dataset.collate
        )
    
    def train_step(self, criterion, model, batch):
        model.train()
        model_out = model(batch)
        loss_out = criterion(**model_out)
        self.aggregate_output(train_loss=loss_out)
        return loss_out

    def valid_step(self, criterion, model, batch, batch_first=False, reduce_func='mean'):
        model.eval()
        with torch.no_grad():
            model_out = model(batch)
            loss_out = criterion(**model_out, train=False)
            predicted = model(batch, decode=True)
        
        res = self.reduce_metrics(**predicted, batch_first=batch_first, reduce=reduce_func)
        self.aggregate_output(valid_loss=loss_out, metrics=res)
        res['loss'] = loss_out['loss'].item(),
        return res
    
    def reduce_metrics(self, decoded_ids, target_ids, batch_first, reduce='mean', **kwargs):
        if not batch_first:
            decoded_ids = decoded_ids.transpose(0, 1)
            target_ids = target_ids.transpose(0, 1)
        decoded_ids = decoded_ids.tolist()
        target_ids = target_ids.tolist()
        decoded_text = [self.voc.decode_line(ids, self.tokenizer.aggregator, remove_eos=True, to_string=True)
            for ids in decoded_ids]
        target_text = [self.voc.decode_line(ids, self.tokenizer.aggregator, remove_eos=True, to_string=True)
            for ids in target_ids]
        res = {
            'decoded_text': decoded_text,
            'target_text': target_text
        }
        return res

    def epoch_end(self, epoch):
        line = f'epoch: {epoch}'
        for k in self.epoch_log_out:
            self.log[k].append(sum(self.epoch_log_out[k]) / len(self.epoch_log_out[k]))
            line += ', {}: {:.4f}'.format(k, self.log[k][-1])
        self.epoch_log_out.clear()
        print(line)

    def aggregate_output(self, train_loss=None, valid_loss=None, metrics=None):
        if train_loss:
            for k in train_loss:
                self.epoch_log_out['train_' + k].append(train_loss[k].item())
        if valid_loss:
            for k in valid_loss:
                self.epoch_log_out['valid_' + k].append(valid_loss[k].item())
        if metrics:
            for k in metrics:
                if k[:4] in ['bleu', 'roug', 'dist']:
                    self.epoch_log_out[k].append(metrics[k])
    
    def print(self, epoch):
        print_line = f'epoch: {epoch}'
        for k in self.epoch_log_out:
            print_line += ', {}: {:.4f}'.format(k, sum(self.epoch_log_out[k]) / len(self.epoch_log_out[k]))
        print(print_line)
    
    def plot_fig(self):
        fig, ax = plt.subplots()
        x = list(range(1, self.config.n_epoch+1))
        for k in self.log:
            y = self.log[k]
            ax.plot(x, y, label=k)
        ax.set_xlabel('epoch')
        ax.set_ylabel('metric')
        ax.legend()
        plt.savefig(f'./log/train_{self.config.model}.png')
    
    def visualize(self, decoded):
        print('decoded text:')
        print(decoded['decoded_text'])
        print('target text:')
        print(decoded['target_text'])


