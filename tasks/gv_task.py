from .base_task import BaseTask

class GVTask(BaseTask):

    def build_criterion(self, config):
        from criterion.kl_loss_and_2CE import KLLossAnd2CE
        loss = KLLossAnd2CE(
            wait_steps=config.wait_steps,
            kl_annealing=config.kl_annealing,
            alpha=config.alpha,
            label_smoothing=config.label_smoothing,
            ignore_idx=self.voc.pad_id
        )
        return loss

    def load_dataset(self, split, separator='\t'):
        return super().load_dataset(split, separator)
    
    def reduce_metrics(
        self, 
        decoded_ids, 
        target_ids, 
        vae_decoded_ids, 
        vae_target_ids, 
        batch_first=False, 
        reduce='mean',
        **kwargs):
        vae_res = super().reduce_metrics(vae_decoded_ids, vae_target_ids, batch_first, reduce)
        for k in list(vae_res.keys()):
            vae_res['vae_' + k] = vae_res[k]
        res = super().reduce_metrics(decoded_ids, target_ids, batch_first, reduce)
        vae_res.update(res)
        vae_res.update(kwargs)
        return vae_res
    
    def visualize(self, decoded):
        super().visualize(decoded)
        print('vae random reconstruction text:')
        print(decoded['vae_decoded_text'])
        print('vae random true text:')
        print(decoded['vae_target_text'])
