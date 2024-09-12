import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv('/Users/erenanbar/Desktop/internship_project/data/ticketData_vs_num_disconnections_202308141804-1692036301198.csv')




bins = np.arange(start = 1,
                 stop = 5,
                 step = 1)
np.append(bins,np.max(df['num_disconnection']))
bin_labels = range(1,len(bins))

df['bin'], bin_edges = pd.cut(df['num_disconnection'], 
                              bins=bins, 
                              labels = bin_labels, 
                              retbins = True)
bin_table = pd.DataFrame(zip(bin_edges, bin_labels, df['bin'].value_counts()),
                            columns=['bin_edge', 'bin_label','value_counts'])
print(bin_table)

df = df.groupby(['modelName','bin']).aggregate({'ticket_size':'sum','total_size':'sum'})
df['call_prob'] = df['ticket_size']/df['total_size']

df.reset_index()
grouped = df.groupby(['modelName'])


model = 'DLink DIR-853/ET'
call_prob = grouped.get_group(model)['call_prob']

plt.plot(bin_labels,call_prob)
plt.xlabel('num_disconnection binned')
plt.ylabel(model+' call_prob')
plt.legend()
plt.show()