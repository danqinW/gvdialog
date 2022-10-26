import os
from torch import nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from metric.word_overlap_metric import distinct, get_corpus_bleu

class Trainer(object):
    def __init__(self, config, n_epoch, task, criterion, model, optimizer, scheduler, cur_epoch, device) -> None:
        self.config = config
        self.n_epoch = n_epoch
        self.task = task
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cur_epoch = cur_epoch
        self.device = device

    @classmethod
    def build_trainer(cls, config):
        if config.task == 'base':
            from tasks import BaseTask as Task
        else:
            from tasks import GVTask as Task
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        config.device = device
        task = Task.setup_task(config)
        task.load_dataset('train', separator=config.separator)
        task.load_dataset('valid', separator=config.separator)
        task.load_dataset('test', separator=config.separator)
        model = task.build_model(config)
        model.init_model()
        criterion = task.build_criterion(config)
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.learning_rate, 
            momentum=0.9,
            nesterov=True
        )
        training_steps = len(task.dataloader['train']) * config.n_epoch
        scheduler = None
        if not config.disable_scheduler:
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                int(training_steps * config.warmup_ratio)
            )

        cur_epoch = 0
        if config.load_ckpt:
            ckpt = torch.load(config.load_ckpt, map_location=torch.device('cpu'))
            model_ckpt = ckpt['model']
            optim_ckpt = ckpt['optim']
            sched_ckpt = ckpt['sched']
            cur_epoch = ckpt['epoch'] + 1
            task.log = ckpt['log']
            model.load_state_dict(model_ckpt)
            optimizer.load_state_dict(optim_ckpt)
            scheduler.load_state_dict(sched_ckpt)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model.to(device)
        print(config)
        return cls(config, config.n_epoch, task, criterion, model, optimizer, scheduler, cur_epoch, device)

    def train(self):
        best_score = None
        while self.cur_epoch < self.n_epoch:
            dataloader = self.task.dataloader['train']
            for batch_i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                self.optimizer.zero_grad()
                batch = self.to_device(batch)
                res = self.task.train_step(self.criterion, self.model, batch)
                res['loss'].backward()
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                if (batch_i + 1) % self.config.evaluate_every == 0:
                    self.validate(reduce_func='mean')
                    self.task.print(self.cur_epoch)
                elif (batch_i + 1) % self.config.print_every == 0:
                    self.task.print(self.cur_epoch)
                
            self.validate(reduce_func='mean')
            self.task.epoch_end(self.cur_epoch)
            if best_score is None or self.task.log['bleu-1'][-1] > best_score:
                best_score = self.task.log['bleu-1'][-1]
                self.save(f'best-{self.config.model}.ckpt', self.cur_epoch)
            last_score = self.task.log['bleu-1'][-1]
            self.save('{}-epoch-{}-bleu-{:.4f}.ckpt'.format(
                self.config.model,
                self.cur_epoch,
                last_score
            ), self.cur_epoch)
            self.cur_epoch += 1
            
        self.task.plot_fig()

    def validate(self, reduce_func='mean'):
        dataloader = self.task.dataloader['valid']
        decoded_text = []
        target_text = []
        metrics = {}
        for batch_i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = self.to_device(batch)
            res = self.task.valid_step(self.criterion, self.model, batch, self.config.batch_first, reduce_func)
            if batch_i == 0:
                self.task.visualize(res)
            decoded_text.extend(res['decoded_text'])
            target_text.extend(res['target_text'])
        preds = [self.task.tokenizer.tokenizer(line) for line in decoded_text]
        tars = [self.task.tokenizer.tokenizer(line) for line in target_text]
        metrics.update(get_corpus_bleu(preds, tars))
        metrics.update(distinct(preds))
        for k in metrics:
            print('{}: {:.4f}'.format(k, metrics[k]), end=', ')
            self.task.epoch_log_out[k].append(metrics[k])
        print()

    def save(self, name, epoch):
        if not os.path.exists(self.config.save_ckpt):
            os.mkdir(self.config.save_ckpt)
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'sched': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'log': self.task.log
        }, os.path.join(self.config.save_ckpt, name))

    def to_device(self, batch):
        train_batch = []
        for t in batch:
            if isinstance(t, torch.Tensor):
                t = t.to(self.device)
            train_batch.append(t)
        return train_batch
