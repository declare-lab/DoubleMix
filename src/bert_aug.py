import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler

class BertAug4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAug4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, inputs, aug_inputs1=None, aug_inputs2=None, lam=None, ws=None, mix_layer=None, attention_mask=None, token_type_ids=None, head_mask=None):

        input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']


        embedding_output = self.embeddings(
            input_ids, token_type_ids=token_type_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0-extended_attention_mask)*-1e4

        head_mask = [None] * self.config.num_hidden_layers

        if aug_inputs1 is not None:
            input_ids1, attention_mask1, token_type_ids1 = aug_inputs1['input_ids'], aug_inputs1['attention_mask'], aug_inputs1['token_type_ids']
            
            extended_attention_mask1 = attention_mask1.unsqueeze(1).unsqueeze(2)
            extended_attention_mask1 = (
                1.0 - extended_attention_mask1) * -10000.0

            embedding_output1 = self.embeddings(
                input_ids1, token_type_ids=token_type_ids1)

        if aug_inputs2 is not None:
            input_ids2, attention_mask2, token_type_ids2 = aug_inputs2['input_ids'], aug_inputs2['attention_mask'], aug_inputs2['token_type_ids'] 
            
            extended_attention_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0
            
            embedding_output2 = self.embeddings(
                input_ids2, token_type_ids=token_type_ids2)

            encoder_outputs = self.encoder(embedding_output, embedding_output1, embedding_output2, lam, ws, mix_layer,
                                           extended_attention_mask, extended_attention_mask1, extended_attention_mask2, 
                                           head_mask=head_mask)
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

    def forward(
        self, 
        hidden_states, 
        hidden_states1=None, 
        hidden_states2=None, 
        lam=None, 
        ws=None, 
        mix_layer=None, 
        attention_mask=None, 
        attention_mask1=None, 
        attention_mask2=None, 
        head_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()

        
        if hidden_states1 is not None:
            if len(ws) == 3:
                noise = 1e-5 * torch.randn_like(hidden_states)
                hidden_states3 = hidden_states + noise

        if mix_layer == -1:
            if hidden_states2 is not None:
                if len(ws) == 3:
                    mix_aug = ws[0] * hidden_states1 + ws[1] * hidden_states2 + ws[2] * hidden_states3
                else:
                    mix_aug = ws[0] * hidden_states1 + ws[1] * hidden_states2
                hidden_states = lam * hidden_states + (1 - lam) * mix_aug

        for i, layer_module in enumerate(self.layer):
            if mix_layer is not None:
                if i<= mix_layer:
                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

                    layer_outputs = layer_module(
                        hidden_states, attention_mask, head_mask[i], output_attentions=self.output_attentions)
                    hidden_states = layer_outputs[0]

                    if self.output_attentions:
                        all_attentions = all_attentions + (layer_outputs[1],)

                    if hidden_states1 is not None:
                        layer_outputs1 = layer_module(hidden_states1, attention_mask1, head_mask[i], output_attentions=self.output_attentions)
                        hidden_states1 = layer_outputs1[0]

                    if hidden_states2 is not None:
                        layer_outputs2 = layer_module(hidden_states2, attention_mask2, head_mask[i], output_attentions=self.output_attentions)
                        hidden_states2 = layer_outputs2[0]

                if i == mix_layer:
                    if hidden_states2 is not None:
                        if len(ws) == 3:
                            mix_aug = ws[0] * hidden_states1 + ws[1] * hidden_states2 + ws[2] * hidden_states3
                        else:
                            mix_aug = ws[0] * hidden_states1 + ws[1] * hidden_states2
                        hidden_states = lam * hidden_states + (1 - lam) * mix_aug

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

        return outputs
        
class Bert_aug(nn.Module):
    def __init__(self, num_labels=3, output_attentions=False):
        super(Bert_aug, self).__init__()
        self.output_attentions = output_attentions
        self.model = BertAug4Mix.from_pretrained('bert-base-uncased', output_attentions=self.output_attentions)          

        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, x1=None, x2=None, lam=None, ws=None, mix_layer=None):
        if self.output_attentions:
            seq_output, pooled_output, attentions = self.model(x, x1, x2, lam, ws, mix_layer)        
        else:          
            seq_output, pooled_output = self.model(x, x1, x2, lam, ws, mix_layer)
            attentions = None
        pooled_output = torch.mean(seq_output, 1)
        predict = self.linear(pooled_output)

        return predict, (pooled_output, attentions)