import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir,'training_metrics.csv')
data = pd.read_csv(data_file,usecols=['epoch','training_loss'])

plt.plot(data['epoch'],data['training_loss'])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')

plt.show()