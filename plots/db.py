import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

samples = np.array([ 
                [8, 22643, 7309, 7564],
                [9, 360248,	120175,	116349],
                [10, 87465,	29442,	27126],
                [11, 39423,	13002,	11619],
                [12, 16198,	5431,	5471],
                [13, 8373,	2745,	2818],           
                [14, 4672,	1569,	1579]
           ])

print(samples[:,1])
  
N = 7
ind = np.arange(N) 
width = 0.25
 
bar1 = plt.bar(ind, samples[:,1], width, color = '#5F9E6E')
bar2 = plt.bar(ind+width, samples[:,2], width, color='#5975A4')
bar3 = plt.bar(ind+width*2, samples[:,3], width, color = '#CC8963')
  
plt.xlabel("k-mers")
plt.ylabel('Num. samples')
#plt.title("Players Score")
  
plt.xticks(ind+width, samples[:, 0] )
plt.legend( (bar1, bar2, bar3), ('Training', 'Validation', 'Testing') )
#plt.show()
#plt.savefig("dataset_samples.png", dpi=300, bbox_inches='tight')

plt.savefig('dataset_samples.eps', format='eps', bbox_inches='tight')

