# Predictions for HLAB dataset for ESM
from model_utils_bert import BertRnn
from transformers import BertConfig
from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from dataloader_bert import DataSetLoaderBERT
from tape import ProteinBertConfig
import pandas as pd

def compute_metrics(pred):
    labels = pred.label_ids
    prediction=pred.predictions
    preds = prediction.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'auc': acc
    }

import argparse

parser = argparse.ArgumentParser(prog='pMHC')
parser.add_argument('-m', '--model', default='results/train_esm2_t6_rnn_freeze_30epochs/checkpoint-202134', help='Path to model')  
parser.add_argument('-i', '--input', default='dataset/hlab/test.csv', help='Input csv')  
parser.add_argument('-o', '--output', default='predictions_output.csv', help='Output csv')  
parser.add_argument('-p', '--pretrained', default='pre_trained_models/esm2_t6_8M_UR50D', help='Pretrained model')  

args = parser.parse_args()
model_name      = args.model
input_name      = args.input
output_name     = args.output
pretrained      = args.pretrained

#model_name = "results/train_esm2_t6_rnn_freeze_30epochs/checkpoint-202134"
seq_length = 50 # for MHC-I
config = BertConfig.from_pretrained(model_name, num_labels=2 )

model = Trainer(model = BertRnn.from_pretrained(model_name, config=config), compute_metrics = compute_metrics)

test = pd.read_csv(input_name)
test["Label"] = 0 # es solo para que funcione el Trainner, xq usando pipeline para las predicciones no funcionaba
test.to_csv("dataset/hlab/test_tmp.csv", index=False)

test_dataset = DataSetLoaderBERT("dataset/hlab/test_tmp.csv", tokenizer_name=pretrained, max_length=seq_length)
predictions, label_ids, metrics = model.predict(test_dataset)

# save predictions ###############################################################
df = pd.DataFrame(predictions)
df = df.rename(columns={0: 'logits_class_0', 1: 'logits_class_1'})

df['prediction'] = df.apply(lambda row: ( 0 if row[0] > row[1] else 1 ), axis=1)
df['peptide'] = test["peptide"]
df['mhc'] = test["mhc"]
df.to_csv(output_name, index=False)
###################################################################################