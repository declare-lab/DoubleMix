import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler

class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, inputs=None,  inputs2=None, lam=None, mix_layer=None, token_type_ids=None, position_ids=None, head_mask=None):

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if inputs2 is not None:
            input_ids2 = inputs2['input_ids']
            attention_mask2 = inputs2['attention_mask']
            token_type_ids2 = inputs2['token_type_ids']

            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)

            #extended_attention_mask2 = extended_attention_mask2.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if inputs2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)

        if inputs2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, lam, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, lam=None, mix_layer=None, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = lam * hidden_states + (1-lam)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if mix_layer is not None:
                if i <= mix_layer:
                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

                    layer_outputs = layer_module(
                        hidden_states, attention_mask, head_mask[i], output_attentions=self.output_attentions)
                    hidden_states = layer_outputs[0]

                    if self.output_attentions:
                        all_attentions = all_attentions + (layer_outputs[1],)

                    if hidden_states2 is not None:
                        layer_outputs2 = layer_module(
                            hidden_states2, attention_mask2, head_mask[i], output_attentions=self.output_attentions)
                        hidden_states2 = layer_outputs2[0]

                if i == mix_layer:
                    if hidden_states2 is not None:
                        hidden_states = lam * hidden_states + (1-lam)*hidden_states2
                
                if i > mix_layer:
                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

                    layer_outputs = layer_module(
                        hidden_states, attention_mask, head_mask[i], output_attentions=self.output_attentions)
                    hidden_states = layer_outputs[0]

                    if self.output_attentions:
                        all_attentions = all_attentions + (layer_outputs[1],)
            else:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i], output_attentions=self.output_attentions)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs
        

class Bert_model(nn.Module):
    def __init__(self, num_labels=3, mix_option=False, output_attentions=False):
        super(Bert_model, self).__init__()
        self.output_attentions = output_attentions
        if mix_option:
            self.model = BertModel4Mix.from_pretrained('bert-base-uncased', output_attentions=self.output_attentions)          
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', output_attentions=self.output_attentions)

        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, x2=None, lam=None, mix_layer=None, mix_option=False):

        if mix_option:
            if self.output_attentions:
                all_hidden, pooler, attentions = self.model(x, x2, lam, mix_layer)  
            else:
                all_hidden, pooler = self.model(x, x2, lam, mix_layer)
                attentions = None
        else:
            bert_output = self.model(**x)
            all_hidden = bert_output.last_hidden_state
            if self.output_attentions:
                attentions = bert_output.attentions
            else:
                attentions = None
        pooled_output = torch.mean(all_hidden, 1)
        predict = self.linear(pooled_output)

        return predict, (pooled_output, attentions)