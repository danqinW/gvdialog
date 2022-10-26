from turtle import forward
from torch import nn

class CrossEntropy(nn.Module):
    def __init__(self, label_smoothing, ignore_idx) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_idx = ignore_idx
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_idx, label_smoothing=label_smoothing)

    def forward(self, output_logits, target_ids, batch_first=False, **kwargs):
        '''
        output_logits: [T, B, C] or [B, T, C] if batch first
        target: [T, B] or [B, T] if batch first
        '''
        if batch_first:
            loss = self.loss(output_logits.permute(0, 2, 1).contiguous(), target_ids)
        else:
            loss = self.loss(output_logits.permute(1, 2, 0).contiguous(), target_ids.T)
        return {'loss': loss}