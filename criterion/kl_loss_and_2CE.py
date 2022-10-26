from turtle import forward
from torch import nn
from .cross_entropy import CrossEntropy

class KLLossAnd2CE(nn.Module):
    def __init__(self, wait_steps=6000, kl_annealing=8000, alpha=1, label_smoothing=0.1, ignore_idx=0) -> None:
        super().__init__()
        self.wait_steps = wait_steps
        self.kl_annealing = kl_annealing
        self.update_step = 1 / kl_annealing
        self.kl_scale = 0
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ignore_idx = ignore_idx

        self.response_loss = CrossEntropy(label_smoothing=label_smoothing, ignore_idx=ignore_idx)
        self.reconstruction_loss = CrossEntropy(label_smoothing=label_smoothing, ignore_idx=ignore_idx)
    
    def forward(
        self, 
        output_logits, 
        target_ids, 
        vae_output_logits, 
        vae_target_ids, 
        z_mu, 
        z_log_var, 
        batch_first=False,
        train=True,
        **kwargs
    ):
        rsp_loss = self.response_loss(output_logits, target_ids, batch_first)
        rcstr_loss = self.reconstruction_loss(vae_output_logits, vae_target_ids, batch_first)
        kl_loss = self.kl_loss(z_mu, z_log_var)
        loss = rsp_loss['loss'] + rcstr_loss['loss'] + self.alpha * self.kl_scale * kl_loss

        if train:
            self.update_annealing_factor()
        return {
            'loss': loss,
            'response_loss': rsp_loss['loss'],
            'reconstruction_loss': rcstr_loss['loss'],
            'kl_loss': kl_loss
        }
    
    def update_annealing_factor(self):
        if self.wait_steps > 0:
            self.wait_steps -= 1
        elif self.kl_scale < 1:
            self.kl_scale += self.update_step

    @staticmethod
    def kl_loss(mu, log_var):
        kl_loss = .5 * (mu.pow(2.0) + log_var.exp() - log_var - 1.0)
        return kl_loss.sum(dim=1).mean()