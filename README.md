# pMHC binding prediction

This a cmd tool for pMHC binding prediction using Transformers and Transfer Learning.

Main Requeriments:
- numpy=1.25.0
- pytorch=1.13
- scikit-learn=1.3.0
- scipy=1.11.1
- matplotlib=3.7.1
- transformers=4.24.0
- tqdm=4.65.0

## Training

In order to train the models you need to run the command: 

```
python train.py -t bert -r results/temp -m models/temp -p pre_trained_models/esm2_t6_8M_UR50D
```

## Making Predictions

For making predictions, run predict.py. Additionally you need to provide:
- Trained model
- Imput: csv with peptide and mhc sequences (dataset/hlab/test.csv is a example)
- Output: output csv
- Pretrained model: this is use for the tokenizer


```
python predict.py -m results/train_esm2_t6_rnn_freeze_30epochs/checkpoint-202134 -i dataset/hlab/test.csv -o predictions_output.csv -p pre_trained_models/esm2_t6_8M_UR50D
```