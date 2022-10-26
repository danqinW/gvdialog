from collections import defaultdict
from dis import dis
from metric.word_overlap_metric import distinct, get_rouge_score
from train import get_config, setup_seed
from base_trainer import Trainer
import pandas as pd
from tqdm import tqdm


config = get_config()
if config.random_seed:
    setup_seed(config.random_seed)

trainer = Trainer.build_trainer(config)
dataloader = trainer.task.dataloader['test']
decoded_text = []
target_text = []
vae_decoded_text = []
vae_target_text = []
contexts = []
metrics = defaultdict(list)
for batch_i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    # rebuild context
    input_var, input_conv_lens, *_ = batch
    input_var = input_var.T.tolist()
    input_conv_lens = input_conv_lens.tolist()
    start = 0
    for l in input_conv_lens:
        context_lines = input_var[start: start+l]
        context_strs = [trainer.task.voc.decode_line(
            line, 
            trainer.task.tokenizer.aggregator, 
            remove_eos=True,
            to_string=True) for line in context_lines]
        contexts.append('===>'.join(context_strs))
        start += l
    # model prediction
    batch = trainer.to_device(batch)
    res = trainer.task.valid_step(trainer.criterion, trainer.model, batch, trainer.config.batch_first, reduce_func=None)
    decoded_text.extend(res['decoded_text'])
    target_text.extend(res['target_text'])
    vae_decoded_text.extend(res['vae_decoded_text'])
    vae_target_text.extend(res['vae_target_text'])
    for k in res:
        if k[:4] in ['bleu', 'roug']:
            metrics[k].extend(res[k])

preds = [trainer.task.tokenizer.tokenizer(line) for line in decoded_text]
tars = [trainer.task.tokenizer.tokenizer(line) for line in target_text]

for k in metrics:
    metrics[k] = sum(metrics[k]) / len(metrics[k])

dist_scores = distinct(preds)
metrics.update(dist_scores)
print_line = ''
for k in metrics:
    print_line += '{}:{:.4f}, '.format(k, metrics[k])
print(print_line)

res = {
    'context': contexts,
    'decoded_text': decoded_text,
    'target_text': target_text,
    'vae_decoded_text': vae_decoded_text,
    'vae_target_text': vae_target_text
}
res.update(metrics)
df = pd.DataFrame(res)
df.to_csv('log/gv-results.csv', index=False)




