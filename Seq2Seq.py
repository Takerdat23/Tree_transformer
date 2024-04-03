import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
from torch.nn import CrossEntropyLoss
from torch.nn import GELU
from modules import *
from transformers import BertModel, BertConfig, T5Config, T5Model, T5Tokenizer,  T5ForConditionalGeneration, AutoTokenizer




class EncoderOutputLayer(nn.Module): 
    """
    (output): Output(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
    """
    def __init__(self, dropout , d_input, d_output):
        super(EncoderOutputLayer, self).__init__()
        self.dense = nn.Linear(d_input, d_output, bias=True)
        self.norm = nn.LayerNorm(d_output, eps=1e-05)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class Dense_Act_Dense(nn.Module): 
    
    def __init__(self,  d_input, d_ff, dropout):
        super(Dense_Act_Dense, self).__init__()
        self.wi = nn.Linear(d_input, d_ff, bias=True)
        self.wo = nn.Linear(d_ff, d_input, bias=True)
        self.dropout  = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm( d_input, eps=1e-05)
 

    def forward(self, x):
        x = self.wi(x)
        x = self.wo(x)
        x = self.dropout(x)
        x = self.layerNorm(x)
   
        return x
    

class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, vocab_size, word_embed, dropout):
        super(Encoder, self).__init__()
        self.word_embed = word_embed
        self.layers = clones(layer, N)
        # self.intermidiate = IntermidiateOutput( d_model, vocab_size)
        # self.output = EncoderOutputLayer(dropout, d_model, d_model)
        
        

    def forward(self, inputs, mask):
        break_probs = []
        hidden_states =[]
    
        x = self.word_embed(inputs)
        group_prob = 0.
        for layer in self.layers:
            x,group_prob,break_prob = layer(x, mask,group_prob)
            hidden_states.append(x)
            break_probs.append(break_prob)

       
        # x= self.intermidiate(x)
       
        break_probs = torch.stack(break_probs, dim=1)
        return x, hidden_states, break_probs


    def masked_lm_loss(self, out, y):
        fn = CrossEntropyLoss(ignore_index=-1)
        return fn(out.view(-1, out.size()[-1]), y.view(-1))


    def next_sentence_loss(self):
        pass

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, vocab_size ,  group_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        # self.selfoutput = SelfOutputLayer(dropout, size, size )
        

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
    
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
 
        x = self.sublayer[1](x, self.feed_forward)

    
        return x, group_prob, break_prob
    
class Encoder_Tree_transfomer(nn.Module): 
    def __init__(self, vocab_size, N=12, d_model=768, d_ff=2048, h=12, dropout=0.1, num_categories= 10, no_cuda= False):
        super(Encoder_Tree_transfomer, self).__init__()
        "Helper: Construct a model from hyperparameters."
        self.no_cuda=  no_cuda
        self.c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
        group_attn = GroupAttention(d_model, no_cuda=self.no_cuda)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, 128)
        word_embed = nn.Sequential(Embeddings(d_model, vocab_size), self.c(position))
        self.encoder = Encoder(EncoderLayer(d_model, self.c(attn), self.c(ff), vocab_size, group_attn, dropout), 
                    N, d_model, vocab_size, self.c(word_embed),  dropout)


        
        

    def forward(self,input_ids,attention_mask):
        _, hiddenStates ,_= self.encoder.forward(input_ids, attention_mask)
        
        # output = self.outputHead.forward(hiddenStates, categories )
        return hiddenStates[-1], input_ids, attention_mask #return last hidden states



class CustomT5Config(T5Config):
    def __init__(self, vocab_size, N=12, d_model=768, d_ff=2048, h=12, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = N
        self.num_heads = h
        self.dropout_rate = dropout
        self.vocab_size = vocab_size 
        self.d_model= d_model
        self.d_ff = d_ff




class Tree_transfomer_5(nn.Module): 
    def __init__(self, vocab_size= 50000, N=12, d_model=768, d_ff=2048, h=12, dropout=0.1, num_categories= 10, no_cuda= False):
        super(Tree_transfomer_5, self).__init__()
        "Helper: Construct a model from hyperparameters."
        self.encoder = Encoder_Tree_transfomer(  vocab_size, N, d_model, d_ff, h, dropout)

        custom_config = CustomT5Config(vocab_size, N, d_model, d_ff, h, dropout)

        # Initialize T5 model with the custom configuration
  
        self.decoder =  T5ForConditionalGeneration(config=custom_config).decoder

        self.lm_head = nn.Linear( d_model, vocab_size, bias=False)
   

      
                

    def forward(self, inputs, mask, labels = None):

        hiddenStates, Enc_input_ids, Enc_attention_masks= self.encoder.forward(inputs, mask)

        output = self.decoder( inputs.long(), mask.long(), 
                              encoder_hidden_states= hiddenStates , 
                              encoder_attention_mask=Enc_attention_masks, 
                              output_attentions=True)
        
        sequence_output = output[0]

      

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
  
        return {
            'loss': loss , 
            'lm_logits': lm_logits, 
            'decoder_hidden_states':  output[0],
            'decoder_attentions': output[2],
            'cross_attentions' : output[3],
        }
device = "cuda"
model = Tree_transfomer_5(vocab_size= 50000, N=12, d_model=768, d_ff=2048, h=12, dropout=0.1).to(device)




tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


test_sentence = "sjgdibhdfigdfigdfbidbj"

test = tokenizer(test_sentence, max_length = 128, truncation = True, padding = "max_length" , return_tensors  ="pt")


# print(test['input_ids'])
input = test['input_ids'].to(device)
attentionmask = test['attention_mask'].to(device)




outputs  = model.forward(input  , attentionmask)
base_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
tokenizer = base_tokenizer.train_new_from_iterator(vsf["train"]["sentence"], vocab_size=2300)
tokenizer.save_pretrained("new_tokenizer")