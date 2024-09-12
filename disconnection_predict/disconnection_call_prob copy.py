import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
df = pd.read_csv('/Users/erenanbar/Desktop/internship_project/data/ticketData_vs_num_disconnections_202308141804-1692036301198.csv')


model_name = 'DLink DIR-853/ET'
#*********************************************************

bins = np.arange(start = 1,
                 stop = 26,
                 step = 5)
      
bin_labels = bins
df['bin'], bin_edges = pd.cut(df['num_disconnection'], 
                              bins=bins, 
                              labels = bin_labels, 
                              retbins = True)
#df containing binning edges corresponding to each bin
bin_table = pd.DataFrame(zip(bin_edges, bin_labels),
                        columns=['bin_edge', 'bin_label'])
print(bin_table.value_counts())
df = df.groupby(['modelName','bin']).aggregate({'ticket_size':'sum','total_size':'sum'}).reset_index()
df['call_prob'] = df['ticket_size']/df['total_size']
df['call_prob'] = df['call_prob'].fillna(0)


isoR = IsotonicRegression(
        y_min=0,
        y_max=1,
        increasing=True,
        out_of_bounds="clip")
grouped = df.groupby(['modelName'])
for model, data in grouped:
    x_train = bin_labels
    y_train = data['call_prob']
    isoR_num_disc = isoR.fit(x_train, y_train)
    train_prediction = isoR_num_disc.predict(x_train)
    df.loc[(df['modelName'] == model[0]), 'prob_isoR'] = train_prediction

    
   
call_prob_predict = grouped.get_group((model_name))['prob_isoR']
call_prob = grouped.get_group((model_name))['call_prob']
plt.plot(bin_labels,call_prob,label= 'call_prob_train')
plt.plot(bin_labels,call_prob_predict,label= 'call_prob_predict')
plt.xlabel('num_disconnection binned')
plt.ylabel(model_name+' call_prob')
plt.legend()
plt.show()

