from torch import nn
import torch.nn.functional as F
import torch
import random

from utils.pad import generate_square_mask
from .base_rnn import *
from utils.registry import register

class RandomContextReconstruction(nn.Module):

    def __init__(self, config, device, embedding) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.embedding = embedding
        self.to_hidden_size = nn.Linear(config.latent_size, config.hidden_size)
        self.gru = nn.GRU(config.d_model, config.hidden_size, num_layers=config.n_layers,
                          dropout=(0 if config.n_layers==1 else config.dropout))
        self.out = nn.Linear(config.hidden_size, config.vocab_size)
        if self.config.share_embedding:
            self.out.weight = self.embedding.weight
    
    def forward(self, trg_var, trg_length, latent_z, decode=False):
        '''
        trg_var: [seq_len, num_sen]
        trg_length: [num_sen]
        latent_z: [num_sen, latent_size]
        '''
        num_sen = trg_var.size(1)
        max_seq_len = trg_length.max().item()
        decoder_input = torch.tensor([[self.config.voc.sos_id for _ in range(num_sen)]]).long().to(self.device)
        # decoder_input = trg_var[:1]
        decoder_hidden = self.to_hidden_size(latent_z)
        decoder_hidden = decoder_hidden.repeat(self.config.n_layers, 1, 1)

        output_logits = []
        all_outputs = []
        for t in range(max_seq_len):
            decoder_input = self.embedding(decoder_input)
            decoder_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            decoder_logits = self.out(decoder_output)
            _, max_i = torch.max(decoder_logits, dim=-1)
            all_outputs.append(max_i.view(1, -1))
            output_logits.append(decoder_logits)
            if decode:
                decoder_input = max_i.view(1, -1)
            else:
                decoder_input = trg_var[t:t+1]
                if random.random() < self.config.word_drop:
                    decoder_input = torch.tensor([[self.config.voc.unk_id for _ in range(num_sen)]]).long().to(self.device)
        if decode:
            res = torch.cat(all_outputs, dim=0)
            return {
                'vae_decoded_ids': res,
                'vae_target_ids': trg_var
            }
        else:
            trg_logits = torch.cat(output_logits, dim=0)
            if self.config.share_embedding:
                factor = self.config.d_model
                trg_logits /= torch.sqrt(torch.tensor(factor, dtype=torch.float32))
            
            return {
                'vae_output_logits': trg_logits,
                'vae_target_ids': trg_var
            }

class VAE(VariationModel):
    def __init__(self, config, device, embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.context_hidden = nn.GRU(config.d_model, config.hidden_size, 
                                     num_layers=config.n_layers, bidirectional=True,
                                     dropout=(0 if config.n_layers == 1 else config.dropout))
        
        self.map_to_feature = FeedForward(2 * config.n_layers * config.hidden_size, config.hidden_size, 
                                num_layers=config.n_layers, hidden_size=config.hidden_size, activation=config.act.upper())

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.to_mu = nn.Linear(config.hidden_size, config.latent_size)
        self.to_log_var = nn.Linear(config.hidden_size, config.latent_size)

        self.random_rec = RandomContextReconstruction(config, device, self.embedding)
    
    def forward(self, encoder_outputs, input_conv_len, rec_input, rec_length, decode=False, deterministic=False, only_z=False):
        sorted_conv_len, indices = torch.sort(input_conv_len, descending=True)
        
        batch_size = input_conv_len.size(0)
        context = encoder_outputs.index_select(1, indices)
        packed = nn.utils.rnn.pack_padded_sequence(context, sorted_conv_len.to('cpu'))
        _, context_hidden = self.context_hidden(packed, None)

        _, inverse_indices = indices.sort()
        context_hidden = context_hidden.index_select(1, inverse_indices)
        context_hidden = context_hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        context_hidden = self.dropout(context_hidden)
        z_feature = self.map_to_feature(context_hidden)
        z_feature = self.layer_norm(z_feature)
        
        # generate latent z: [batch_size, latent_size]
        z_mu = self.to_mu(z_feature)
        z_log_var = self.to_log_var(z_feature)
        z = self.connect(z_mu, z_log_var, deterministic)
        # kl_loss = self.kl_divergence(z_mu, z_log_var)
        res = {
            'z': z,
            'z_mu': z_mu,
            'z_log_var': z_log_var
        }
        if only_z:
            return res

        # random reconstruction
        vae_res = self.random_rec(rec_input, rec_length, z, decode=decode)
        res.update(vae_res)
        return res

@register('gvdialog')
class GVDialog(VariationModel):
    def __init__(self, config, model_type='recosa') -> None:
        super().__init__()

        self.config = config
        self.device = config.device
        self.model_type = model_type

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pe = nn.Embedding(config.max_seq_len, config.d_model)
        self.word_enc = EncoderRNN(config, self.device)
        self.ctx_enc = ContextEncoder(config, self.device)
        self.vae = VAE(config, self.device, self.embedding)
        self.to_dmodel_size = FeedForward(config.latent_size, config.d_model * config.chunk_size, num_layers=1,
                            hidden_size=config.hidden_size, activation=config.act.upper())
        # self.to_dmodel_size = nn.Linear()
        self.layer_norm = nn.LayerNorm(config.d_model)
        # self.dec = GVDialogDecoder(config, device)
        decoder_layer = nn.TransformerDecoderLayer(
            config.d_model,
            config.n_head,
            config.dim_feedforward,
            config.dropout,
            activation=config.act,
            norm_first=self.config.norm_first
        )
        self.dec = nn.TransformerDecoder(decoder_layer, config.num_decoder_blocks)
        self.dec_dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.d_model, config.vocab_size)

        self.scale_factor = 1 / torch.sqrt(torch.tensor(config.d_model, dtype=torch.float32))
        if config.share_embedding:
            self.out.weight = self.embedding.weight
    
    def init_model(self):
        # Initialize parameters
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if name.split('.')[0] in ['embedding', 'pe']:
                    nn.init.normal_(param)
                    if name.split('.')[0] == 'embedding':
                        nn.init.constant_(param[self.config.voc.pad_id], 0)
                else:
                    nn.init.xavier_uniform_(param)

    def forward(self, batch, decode=False, only_z=False, deterministic=False):
        # encoder_outputs: [context_len, batch_size, d_model]
        input_var, input_conv_lens, input_sen_lens, output_var, _, max_output_len = batch
        word_input = self.embedding(input_var)
        sen_repr = self.word_enc(word_input, input_sen_lens)
        # encoder_outputs, encoder_mask = self.enc(input_var, input_conv_lens, input_sentence_lens)

        max_conv_len = torch.max(input_conv_lens).item()
        pos_emb = self.pe(torch.arange(0, max_conv_len).long().view(-1, 1).to(self.device))
        encoder_outputs, encoder_mask, conv_repr = self.ctx_enc(sen_repr, input_conv_lens, pos_emb)
        
        rec_input, rec_length = self.sample_random_context(input_var, input_conv_lens, input_sen_lens)
        vae_res = self.vae(conv_repr, input_conv_lens, rec_input, rec_length, decode, deterministic, only_z)
        if only_z:
            return vae_res
        # vae_res = {}
        # response generation

        z = vae_res['z']
        latent_z = self.to_dmodel_size(z)
        latent_z = self.layer_norm(latent_z)
        batch_size = output_var.size(1)
        # begin_tokens = torch.tensor([[self.config.voc.sos_id for _ in range(batch_size)]]).long().to(self.device)
        
        if not decode:
            tgt_len = max_output_len + 1 if self.config.use_latent else max_output_len
            seq_mask = generate_square_mask(tgt_len).to(self.device)

            trg_input = output_var[:-1]
            pad_mask = (trg_input.T == self.config.voc.pad_id)
            trg_input = self.embedding(trg_input)
            trg_input += self.pe(torch.arange(0, max_output_len).long().view(-1, 1).to(self.device))
            if self.config.use_latent:
                trg_input = torch.cat([latent_z.unsqueeze(0), trg_input], dim=0)
                unk = torch.tensor([[self.config.voc.unk_id for _ in range(batch_size)]]).long().to(self.device)
                pad_mask = (torch.cat([unk, output_var[:-1]]).T == self.config.voc.pad_id)
            trg_input = self.dec_dropout(trg_input)
            trg_output = self.dec(trg_input, encoder_outputs, seq_mask, 
                        tgt_key_padding_mask=pad_mask, memory_key_padding_mask=encoder_mask)

            trg_logits = self.out(trg_output)
            if self.config.use_latent:
                trg_logits = trg_logits[1:]
            vae_res['output_logits'] = trg_logits

        else:
            trg_input = output_var[:1]
            tgt_len = self.config.max_seq_len + 1 if self.config.use_latent else self.config.max_seq_len
            seq_mask = generate_square_mask(tgt_len).to(self.device)
            for t in range(self.config.max_seq_len):
                idx = t + 2 if self.config.use_latent else t + 1
                trg_input_emb = self.embedding(trg_input)
                trg_input_emb += self.pe(torch.arange(0, t + 1).view(-1, 1).to(self.device))
                if self.config.use_latent:
                    trg_input_emb = torch.cat([latent_z.unsqueeze(0), trg_input_emb], dim=0)
                    
                trg_input_emb = self.dec_dropout(trg_input_emb)
                trg_output = self.dec(trg_input_emb, encoder_outputs, seq_mask[:idx, :idx], 
                    memory_key_padding_mask=encoder_mask)
                
                trg_logits = self.out(trg_output)
                _, max_i = torch.max(trg_logits[-1], dim=-1)
                
                trg_input = torch.cat((trg_input, max_i.view(1, -1)), dim=0)
            decoded_batch = trg_input[1:]
            vae_res['decoded_ids'] = decoded_batch
        vae_res['target_ids'] = output_var[1:]
        return vae_res

    def sample_random_context(self, input_var, input_conv_len, input_sentence_length):
        '''
        input_var: [batch_size, num_sen]
        '''
        with torch.no_grad():
            batch_size = input_conv_len.size(0)
            select_indices = torch.ones(batch_size).long().to(self.device) * -1
            start = 0
            for i, l in enumerate(input_conv_len.tolist()):
                input_length = input_sentence_length[start: start+l]
                sample_p = input_length / input_length.sum()
                ind = torch.multinomial(sample_p, 1, replacement=True)
                select_indices[i] = start + ind[0]
                start += l
            select_indices = select_indices.to(self.config.device)
            rec_input = torch.index_select(input_var, 1, select_indices)
            rec_length = input_sentence_length[select_indices]
            max_length = rec_length.max().item()
            rec_input = rec_input[:max_length, :]
        return rec_input, rec_length

