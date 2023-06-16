import pandas as pd
import numpy as np
import os
import re

# esta implementacion trataba de agrupar por allele
"""
data = pd.read_csv('dataset/hlab/hlab_test_micro.csv')
hla_list = data.HLA.unique()
data = data.sort_values(by=['HLA'], ascending=True)
data = data.set_index(['HLA'])
#print( data)
#print(data.loc["HLA-A*01:01"]["mhc"])
#print(data.loc[hla]["peptide"])

for hla in hla_list:
    # peptides
    hla_peptide = data.loc[hla]["peptide"]
    print("HLA:", hla)    

    # save peptides
    #hla_peptide.to_csv("netMHCpan-4.1/test/tmp_peptides.pep", index=False, header=False)
    hla_peptide.to_csv("tmp_peptides.pep", index=False, header=False)

    #cmd = "cd netMHCpan-4.1/test; tcsh ../netMHCpan.sh -a " + hla.replace("*", "") + " -p tmp_peptides.pep -hlapseudo /home/vicente/projects/pmhc/dataset/hlab/MHC_pseudo.dat > tmp_results_netMHCpan4.1"
    cmd = "tcsh netMHCpan-4.1/netMHCpan.sh -a " + hla.replace("*", "") + " -p tmp_peptides.pep -hlapseudo /home/vicente/projects/pmhc/dataset/hlab/MHC_pseudo.dat -xls -xlsfile tmp_results.xls"
    print(cmd)
    os.system(cmd)

    #res = pd.read_excel("tmp_results.xls", sheet_name='Sheet1')
    #print(res)
    break

    # read results
"""

data = pd.read_csv('dataset/hlab/hlab_test.csv')
#print( data)
final_data_results = []

for index, row in data.iterrows():
    # prepare peptides
    pep = open("tmp_peptides.pep", 'w')
    pep.write(row['peptide'])
    pep.close()
    
    cmd = "tcsh netMHCpan-4.1/netMHCpan.sh -a " + row['HLA'].replace("*", "") + " -p tmp_peptides.pep -hlapseudo /home/vicente/projects/pmhc/dataset/hlab/MHC_pseudo.dat > tmp_results.txt"
    #print(cmd)
    os.system(cmd)

    # generate results
    result = open("tmp_results.txt", 'r')
    result_lines = result.readlines()
    result.close()

    # parse results
    row_data = result_lines[-6]
    # possible row_data 
    # 1 HLA-A*01:01       TPDIKSHY TPD-IKSHY  0  0  0  3  1     TPDIKSHY         PEPLIST 0.0872910    1.400 <= WB
    # 1 HLA-A*01:01       LFGRDLSY -LFGRDLSY  0  0  0  0  1     LFGRDLSY         PEPLIST 0.0162280    3.731
    
    #print( row_data )
    #print( re.sub('\s+', ' ', row_data.strip()))
    # reemplzamoa los white spaces por un solo whitespace
    row_data = re.sub('\s+', ' ', row_data.strip())    
    cols_data = row_data.split(" ")
 
    if cols_data[-1].strip() == "SB" or cols_data[-1].strip() == "WB" :
        el_rank = [cols_data[1], cols_data[2], cols_data[-3], cols_data[-1].strip(), 1]
    else:
        el_rank = [cols_data[1], cols_data[2], cols_data[-1].replace("\n", ""), "-", 0]

    final_data_results.append(el_rank)
  
    if index%1000 == 0:
        print("index", index)


df = pd.DataFrame(np.array(final_data_results), columns = ['mhc', 'peptide', 'rank','ligand', 'prediction'])
df.to_csv("predictions_netmhcpan4.1.csv")
print(df)