from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate import bleu_score
from nltk import ngrams
from rouge import Rouge
from collections import Counter
import numpy as np

cherry = bleu_score.SmoothingFunction()
bleu_names = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']
def get_bleu_scores(predicted, target, n_gram=4, reduce='mean'):
    bleus = bleu_names[:n_gram]
    # bleu score
    references = [[ref] for ref in target]
    candidates = predicted
    weights = []
    for i in range(1, n_gram+1):
        weights.append(tuple(1 / i for _ in range(i)))
    scores = [[sentence_bleu(refs, cand, weights=weights[i], smoothing_function=cherry.method1) for i in range(n_gram)]
        for refs, cand in zip(references, candidates)]
    scores = list(zip(*scores))
    if reduce == 'mean':
        scores = [sum(l) / len(l) for l in scores]
    res = dict(zip(bleus, scores))
    res['bleu'] = res['bleu-1']
    return res

def get_corpus_bleu(predicted, target, n_gram=4):
    bleus = bleu_names[:n_gram]
    # bleu score
    references = [[ref] for ref in target]
    candidates = predicted
    scores = []
    for i in range(1, n_gram+1):
        weights = np.ones(i) * (1 / i)
        s = corpus_bleu(references, candidates, weights=weights, smoothing_function=cherry.method4)
        scores.append(s)
    return dict(zip(bleus, scores))

def get_rouge_score(predicted, target, reduce='mean'):
    pred_true_pairs = list(zip(predicted, target))
    keep_pairs = [pair for pair in pred_true_pairs if len(pair[0]) > 0 and pair[0][0] not in ['.']]
    if len(keep_pairs) == 0: return {}
    print(len(keep_pairs) / len(pred_true_pairs))
    pred, targ = list(zip(*keep_pairs))
    pred = [' '.join(l) for l in pred]
    targ = [' '.join(l) for l in targ]
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hyps=pred, refs=targ, avg=True if reduce == 'mean' else False)
    if reduce == 'mean':
        res = {k: v['r'] for k, v in rouge_scores.items()}
    else:
        res = {}
        res['rouge-1'] = [elem['rouge-1']['r'] for elem in rouge_scores]
        res['rouge-2'] = [elem['rouge-2']['r'] for elem in rouge_scores]
        res['rouge-l'] = [elem['rouge-l']['r'] for elem in rouge_scores]
    return res

def distinct_n_sentence(sentence, n):
    if len(sentence) == 0:
        return 0.0
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def get_distinct_score(preds, n):
    return sum(distinct_n_sentence(s, n) for s in preds) / len(preds)

# calculate Distinct-1/2 for dailydialog & personachat
def distinct(pred_sentences) -> dict:
    
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for hyp in pred_sentences:
        unigrams = Counter(hyp)
        bigrams = Counter(zip(hyp, hyp[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(hyp)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(hyp)-1)+1e-5))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    # output_content = 'Distinct-1/2: {}/{}\n'.format(round(inter_dist1, 4), round(inter_dist2, 4))
    # print('-------------- Distinct score --------------\n{}'.format(output_content))
    return {
        'intra_dist1': intra_dist1,
        'intra_dist2': intra_dist2,
        'inter_dist1': inter_dist1,
        'inter_dist2': inter_dist2
    }