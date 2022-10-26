from torch import nn
import torch.nn.functional as F
import torch

class EncoderRNN(nn.Module):

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=config.n_layers, bidirectional=True, 
                          dropout=(0 if config.n_layers == 1 else config.dropout))
        self.gru_output_size = 2 * config.n_layers * config.hidden_size
        self.to_dmodel_size = FeedForward(self.gru_output_size, config.d_model, num_layers=1, 
                            hidden_size=config.hidden_size, activation=config.act.upper())
        self.activate = getattr(nn, config.act.upper())()
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_var, input_lengths, hidden=None):
        '''
            input_var: [seq_len, all_sen_len, hidden_size]
            input_length: [all_sen_len]
        '''
        sorted_length, indices = torch.sort(input_lengths, descending=True)
        input_var = input_var.index_select(1, indices)

        packed = nn.utils.rnn.pack_padded_sequence(input_var, sorted_length.to('cpu'))
        encoder_outputs, encoder_hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)

        _, reverse_indices = indices.sort()
        outputs = outputs.index_select(1, reverse_indices)
        encoder_hidden = encoder_hidden.index_select(1, reverse_indices)

        num_sen = encoder_hidden.size(1)
        encoder_hidden = encoder_hidden.transpose(0, 1).contiguous().view(num_sen, -1)
        sent_repr = self.activate(self.to_dmodel_size(encoder_hidden))
        sent_repr = self.layer_norm(sent_repr)
        return sent_repr

class ContextEncoder(nn.Module):

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        
        self.dropout = nn.Dropout(config.dropout)
        trf_encoder_layer = nn.TransformerEncoderLayer(
                                config.d_model,
                                config.n_head,
                                config.dim_feedforward,
                                config.dropout,
                                activation=config.act,
                                norm_first=self.config.norm_first,            
                            )
        self.trf = nn.TransformerEncoder(trf_encoder_layer, num_layers=config.num_encoder_blocks)

    def forward(self, trf_input_hidden, input_conv_len, pos_emb):
        '''
        input_var: [seq_len, num_seq]
        input_conv_len: [batch_size]
        input_sen_len: [num_sen]
        '''

        max_conv_len = torch.max(input_conv_len).item()
        conv_mask = torch.zeros(len(input_conv_len), max_conv_len).byte().to(self.device)
        start = 0
        trf_input = []
        for i, l in enumerate(input_conv_len.tolist()):
            context = torch.cat((trf_input_hidden[start: start + l], 
                                 torch.zeros(max_conv_len - l, self.config.d_model).to(self.device)), dim=0)
            trf_input.append(context)
            conv_mask[i][l:] = 1
            start += l
        
        conv_mask = conv_mask.bool()
        
        conv_repr = torch.stack(trf_input, dim=1)
        trf_input = conv_repr  + pos_emb
        trf_input = self.dropout(trf_input)
        trf_output = self.trf(trf_input, src_key_padding_mask=conv_mask)

        return trf_output, conv_mask, conv_repr

class EncoderTransformerRNN(nn.Module):

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.dropout = nn.Dropout(config.dropout)
        trf_encoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_head,
            config.dim_feedforward,
            config.dropout,
            activation=config.act,
            norm_first=False,
        )
        self.trf = nn.TransformerEncoder(trf_encoder_layer, num_layers=config.n_layers)
        self.gru = nn.GRU(config.d_model, config.hidden_size, num_layers=1, bidirectional=True)

        self.gru_output_size = 2 * config.d_model
        self.to_context = FeedForward(self.gru_output_size, config.d_model, num_layers=1, 
                            hidden_size=config.hidden_size, activation=config.act.upper())
        self.activate = getattr(nn, config.act.upper())()
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, input_var, input_length, hidden=None):
        '''
        input_var: [max_seq_len, num_sen, d_model]
        input_length: [num_sen]
        '''
        max_seq_len, num_sen = input_var.size(0), input_var.size(1)
        input_var = self.dropout(input_var)
        pad_mask = torch.arange(1, max_seq_len + 1).view(1, -1).to(self.device) > input_length.view(-1, 1)
        output = self.trf(input_var, src_key_padding_mask=pad_mask)

        sorted_length, indices = input_length.sort(descending=True)
        output = output.index_select(1, indices)

        packed = nn.utils.rnn.pack_padded_sequence(output, sorted_length.to('cpu'))
        encoder_outputs, encoder_hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)

        _, reverse_indices = indices.sort()
        outputs = outputs.index_select(1, reverse_indices)
        hiddens = encoder_hidden.index_select(1, reverse_indices)

        hiddens = hiddens.transpose(0, 1).contiguous().view(num_sen, -1)
        trf_input_hidden = self.activate(self.to_context(hiddens))
        sent_repr = self.layer_norm(trf_input_hidden)
        return sent_repr

class VariationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def re_parameterize(mu, log_var):
        # import torch
        # log_var = torch.zeros(4)
        # sampled latent variable using re-parameterize trick
        std = log_var.mul(.5).exp()
        eps = torch.zeros_like(std).normal_()
        return mu + torch.mul(eps, std)

    def connect(self, mu, log_var, deterministic):
        # during inference, sampled process is deterministic if variance equals to zero
        if deterministic:
            log_var.fill_(.0)
        # re-parameterization
        z = self.re_parameterize(mu, log_var)
        return z
    
    @staticmethod
    def kl_divergence(mu, log_var):
        # calculate k-l loss need to be optimized
        # cited papers:
        # 1. generating sentences from a continuous space
        # 2. improving variational inference with inverse autoregressive flow
        # 3. improving variational encoder-decoders in dialogue generation
        # 4. learning discourse-level diversity for neural dialog models using conditional variational auto-encoders
        kl_loss = .5 * (mu.pow(2.0) + log_var.exp() - log_var - 1.0)
        return kl_loss.sum(dim=1)

class GVDialogDecoder(nn.Module):

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.trf_decoder = nn.ModuleList([nn.TransformerDecoderLayer(
            config.d_model,
            config.n_head,
            config.dim_feedforward,
            config.dropout,
            activation=config.act,
            norm_first=False
        ) for _ in range(config.num_decoder_blocks)])
        
    
    def forward(self, trg_input, tgt_seq_mask, tgt_pad_mask, encoder_outputs, encoder_mask, latent_z=None):
        '''
        trg_input: [trg_len, batch_size, d_model]
        encoder_outputs: [max_conv_len, batch_size, d_model]
        latent_z: [batch_size, d_model]
        '''
       
        latent_z = latent_z.unsqueeze(0)
        for layer in self.trf_decoder:
            if latent_z is not None:
                trg_input = torch.cat([latent_z, trg_input], dim=0)
            layer_output = layer(trg_input, encoder_outputs, tgt_seq_mask,
                                 tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=encoder_mask)
            trg_input = layer_output[1:]

        trg_output = trg_input
        return trg_output


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=None,
                 activation="GELU", bias=True):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)()
        n_inputs = [input_size] + [hidden_size] * (num_layers - 1)
        n_outputs = [hidden_size] * (num_layers - 1) + [output_size]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias)
                                      for n_in, n_out in zip(n_inputs, n_outputs)])

    def forward(self, input):
        x = input
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)

        return x