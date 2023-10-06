# read test data
# se crea otro codigo porque ACME usa TF 1.5 y otras librerias antiguas, entonces hemos creado otro ambiente

import pandas as pd
import numpy as np
import os

data = pd.read_csv("../dataset/hlab/hlab_test2.csv")
print(data.head(2))

# group by peptide length (k-mer)
data_grouped = []  # cada elemento es un dataframe. 
# Ex: data_list[0] contiene los datos de peptidos de longitud 8
# Ex: data_list[1] contiene los datos de peptidos de longitud 9

for i in range(8,15):
    peptides = data[data['Length'] == i] 
    data_grouped.append(peptides)
    #peptides.to_csv(str(i) + "-mer.csv", index=False)

print(data_grouped[0].head(3)) # 8-mer peptides


# por cada grupo k-mer, ahora separamos por el tipo de HLA
import os
import glob

def predict_acme(hla, peptides):
    # save peptides in a tmp file
    tmp_peptide_file = open("ACME/ACME_codes/binding_prediction/prediction_input.txt", "w")
    tmp_peptide_file.write(("\t"+hla.replace("HLA-", "")+"\n").join(peptides))
    tmp_peptide_file.write("\t"+hla.replace("HLA-", "")+"\n") # el ultimo petido estaba incompleto
    tmp_peptide_file.close()

    # cmd for anthem
    #python sware_b_main.py --length 9 --HLA HLA-A*01:01 --mode prediction --peptide_file test/predictpeptide.txt

    os.remove("ACME/ACME_codes/results/binding_prediction.txt")
    cmd = "cd ACME/ACME_codes; python binding_prediction.py"    
    os.system(cmd)          
    
    acme_results = open("ACME/ACME_codes/results/binding_prediction.txt", "r")
    lines = acme_results.readlines() 
    bindings = []
    probs = []
    for line in lines: # procesamos cada peptido         
        tmp = line.split("\t")   
        peptide = tmp[0] # aqui esdta el peptido
        prob = float(tmp[2].strip().replace("\n", "")) # aqui esta la probabilidad
        binding = 1 if prob > 0.42 else 0 # segun el github de ACME: https://github.com/HYsxe/ACME    
        bindings.append(binding)
        probs.append(prob)
    return bindings , probs  
    

  
final_results = pd.DataFrame(columns=['id','HLA','peptide','Label','Length','mhc','acme_pred', 'acme_prob'])
for i, df_by_kmer in enumerate(data_grouped): # itereamos por cada k-mer
    print("predicting k-mer ...", i)
    hlas = df_by_kmer['HLA'].unique()



    for hla in hlas:
        print("HLA:", hla )
        df_by_hla = df_by_kmer[df_by_kmer['HLA'] == hla] # dividimos por tipo de hla
        peptides = df_by_hla["peptide"].tolist() 

        if hla[4] == "C" or hla == "HLA-A*02:50" or hla == "HLA-A*24:06" or hla == "HLA-A*24:13" or hla == "HLA-A*32:15" or hla == "HLA-B*45:06" or hla == "HLA-B*83:01": # acme no tiene pseusecuencias para HLA-C
            bindings = np.zeros(len(peptides))
            probs = np.zeros(len(peptides))
        else:            
            bindings, probs = predict_acme(hla, peptides)           
        df_by_hla["acme_pred"] = bindings
        df_by_hla["acme_prob"] = probs
        final_results = pd.concat([final_results, df_by_hla]) 
        print("\n\n")
        #break

    #break
    

final_results = final_results.sort_values('id')
final_results.to_csv("prediction_acme_full.csv", index=False)
print("finish :)")