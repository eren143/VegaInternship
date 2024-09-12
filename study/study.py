import pandas as pd
import numpy as np
array = np.array([[1,2,3],
                 [10,20,30],
                 [500,432,65]]
                 )
df = pd.DataFrame(array,index=[1,2,3],columns=['first','second','third'])
print(df)
df['second'] = df['first'] + df['third']
print(df)