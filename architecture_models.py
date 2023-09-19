# this script analyzed architecture of TAPE, ESM2 and ProtBERT

from transformers import Trainer, TrainingArguments, BertConfig, AdamW
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt, BertRnnSigmoid
from model_utils_tape import TapeLinear, TapeRnn, TapeRnnAtt
from tape import ProteinBertConfig
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from transformers import set_seed
import sys      
#######################################################################
# tape
'''
model_name = "bert-base"   # TAPE   
config = ProteinBertConfig.from_pretrained(model_name, num_labels=2) 
config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = 50
config.cnn_filters = 512
config.cnn_dropout = 0.1   

model_ = TapeRnn.from_pretrained(model_name, config=config)
#print(model_)
#sys.exit()

##################################################################

# esm2 and protbert-bfd #########################################
model_name = "pre_trained_models/esm2_t6_8M_UR50D"          # train 
#model_name = "pre_trained_models/esm2_t12_35M_UR50D" 
#model_name = "pre_trained_models/esm2_t30_150M_UR50D"
#model_name = "pre_trained_models/esm2_t33_650M_UR50D"       # 
#model_name = "pre_trained_models/prot_bert_bfd"
config = BertConfig.from_pretrained(model_name, num_labels=2)
config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = 50
config.cnn_filters = 512
config.cnn_dropout = 0.1  
model_ = BertRnn.from_pretrained(model_name, config=config)
#print(model_)
'''
##################################################################

#import torchvision
#from torchview import draw_graph
#model_graph = draw_graph(model_, input_size=(16,50), expand_nested=True)
#model_graph.visual_graph

from transformers import AutoModel, AutoTokenizer
from torchview import draw_graph
model1 = AutoModel.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")

model_graph = draw_graph(model1, input_data=inputs)

model_graph.visual_graph




