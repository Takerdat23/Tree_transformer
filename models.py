import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
from torch.nn import CrossEntropyLoss
from torch.nn import GELU
from modules import *
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class Topic_SA_Output(nn.Module): 
    def __init__(self, d_input, topic_output, sentiment_output):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(Topic_SA_Output, self).__init__()
        self.Topicdense = nn.Linear(d_input , topic_output,  bias=True)
        self.SentimentDense = nn.Linear(d_input , sentiment_output,  bias=True)
     
   
    

    def forward(self, model_output ):
        """ 
         x : Model output 
         categories: aspect, categories  
         Output: sentiment output 
        """
        pooled_output = model_output[-1][: , 0 , :]

        topic = self.Topicdense(pooled_output )

        sentiment = self.SentimentDense(pooled_output)

        return topic , sentiment




    
class Aspect_Based_SA_Output(nn.Module): 
    def __init__(self, dropout , d_input, d_output, num_categories):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(Aspect_Based_SA_Output, self).__init__()
        self.dense = nn.Linear(d_input * 4 , d_output *num_categories ,  bias=True)
        # self.softmax = nn.Softmax(dim=-1) 
        self.norm = nn.LayerNorm(d_output, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.num_categories = num_categories
        self.num_labels= d_output

    def forward(self, model_output ,categories ):
        """ 
         x : Model output 
         categories: aspect, categories  
         Output: sentiment output 
        """
        pooled_output = torch.cat([model_output[i] for i in range(-4, 0)], dim=-1)[: , 0 , :]

      
      
        x = self.dropout(pooled_output)
        output = self.dense(x)
        # Reshape the output to match the required dimensions
        output = output.view(-1, self.num_categories, self.num_labels)
        return output





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

class IntermidiateOutput(nn.Module): 
    """
    (intermediate): RobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
    )
    """
    def __init__(self,  d_input, vocab_size):
        super(IntermidiateOutput, self).__init__()
        self.dense = nn.Linear(d_input, vocab_size, bias=True)
        self.intermediate_act_fn = GELU()
 

    def forward(self, x):
        x = self.dense(x)
        x = self.intermediate_act_fn(x)
   
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, vocab_size, word_embed, dropout):
        super(Encoder, self).__init__()
        self.word_embed = word_embed
        self.layers = clones(layer, N)
        self.intermidiate = IntermidiateOutput( d_model, vocab_size)
        self.output = EncoderOutputLayer(dropout, d_model, d_model)
        
        

    def forward(self, inputs, mask):
        break_probs = []
        hidden_states =[]
    
        x = self.word_embed(inputs)
        group_prob = 0.
        for layer in self.layers:
            x,group_prob,break_prob = layer(x, mask,group_prob)
            hidden_states.append(x)
            break_probs.append(break_prob)

       
        x= self.intermidiate(x)
       
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

class Constituent_Pretrained_transfomer(nn.Module): 
    def __init__(self, vocab_size, N=12, d_model=768, d_ff=2048, h=12, dropout=0.1, num_categories= 10, no_cuda= False):
        super(Constituent_Pretrained_transfomer, self).__init__()
        "Helper: Construct a model from hyperparameters."
        self.no_cuda=  no_cuda
        self.c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
        group_attn = GroupAttention(d_model, no_cuda=self.no_cuda)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, 128)
        self.word_embed = nn.Sequential(Embeddings(d_model, vocab_size), self.c(self.position))
        self.constituent_Module= EncoderLayer(d_model, self.c(attn), self.c(ff), vocab_size, group_attn, dropout) 
        self.encoder = AutoModel.from_pretrained("vinai/phobert-base").encoder 

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.outputHead = Aspect_Based_SA_Output(dropout , d_model, 4, num_categories ) # 4 class label
    
    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = float
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 ):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

        
        

    def forward(self, inputs, mask, categories):
        x = self.word_embed(inputs)

        

        
        group_prob = 0.
        x,group_prob,break_prob = self.constituent_Module(x, mask,group_prob)

       

        input_shape = inputs.size()
        mask = mask.double()
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(mask, input_shape, dtype = mask.dtype)
        


        x = self.encoder(x , extended_attention_mask.float(), output_hidden_states = True)


        
        output = self.outputHead.forward(x.hidden_states , categories )
        return output



    
class ABSA_Tree_transfomer(nn.Module): 
    def __init__(self, vocab_size, N=12, d_model=768, d_ff=2048, h=12, dropout=0.1, num_categories= 10, no_cuda= False):
        super(ABSA_Tree_transfomer, self).__init__()
        "Helper: Construct a model from hyperparameters."
        self.no_cuda=  no_cuda
        self.c = copy.deepcopy
        self.attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
        self.group_attn = GroupAttention(d_model, no_cuda=self.no_cuda)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, 128)
        self.word_embed = nn.Sequential(Embeddings(d_model, vocab_size), self.c(self.position))
        self.encoder = Encoder(EncoderLayer(d_model, self.c(self.attn), self.c(self.ff), vocab_size, self.group_attn, dropout), 
                    N, d_model, vocab_size, self.c(self.word_embed),  dropout)
        self.outputHead = Aspect_Based_SA_Output(dropout , d_model, 4, num_categories ) # 4 class label

        
        

    def forward(self, inputs, mask, categories):
        _, hiddenStates ,_= self.encoder.forward(inputs, mask)
        
        output = self.outputHead.forward(hiddenStates, categories )
        return output


#Base transformer

class BaseEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, vocab_size ,  dropout):
        super(BaseEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        
        self.size = size
        # self.selfoutput = SelfOutputLayer(dropout, size, size )
        

    def forward(self, x, mask):
  
    
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,  mask= mask))
 
        x = self.sublayer[1](x, self.feed_forward)

    
        return x



class BaseEncoder(nn.Module):
    def __init__(self, layer, N, d_model, vocab_size, word_embed, dropout):
        super(BaseEncoder, self).__init__()
        self.word_embed = word_embed
        self.layers = clones(layer, N)
        self.intermidiate = IntermidiateOutput( d_model, vocab_size)
        self.output = EncoderOutputLayer(dropout, d_model, d_model)
        
        

    def forward(self, inputs, mask):
    
        hidden_states =[]
    
        x = self.word_embed(inputs)

        for layer in self.layers:
            x = layer(x, mask)
            hidden_states.append(x)
        

        x= self.intermidiate(x)
      
       
   
        return x, hidden_states


    def masked_lm_loss(self, out, y):
        fn = CrossEntropyLoss(ignore_index=-1)
        return fn(out.view(-1, out.size()[-1]), y.view(-1))


    def next_sentence_loss(self):
        pass


class ABSA_transfomer(nn.Module): 
    def __init__(self, vocab_size, N=12, d_model=768, d_ff=2048, h=12, num_categories = 10 ,  dropout=0.1, no_cuda= False):
        super(ABSA_transfomer, self).__init__()
        "Helper: Construct a model from hyperparameters."

        self.no_cuda=  no_cuda
        self.c = copy.deepcopy
        self.attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, 128)
        self.word_embed = nn.Sequential(Embeddings(d_model, vocab_size), self.c(self.position))
        self.encoder = BaseEncoder(BaseEncoderLayer(d_model, self.c(self.attn), self.c(self.ff), vocab_size, dropout), 
                    N, d_model, vocab_size, self.c(self.word_embed),  dropout)
        self.outputHead = Aspect_Based_SA_Output(dropout , d_model, 4, num_categories ) # 4 class label

        
        

    def forward(self, inputs, mask, categories):
        _, hiddenStates= self.encoder.forward(inputs, mask)
      
        output = self.outputHead.forward(hiddenStates, categories)
        return output
    

