#%%
import pandas as pd

from sklearn.metrics import confusion_matrix

#%%
results_df = pd.read_csv('transformed_data/results_on_validation.csv', index_col=0)
results_targets = results_df[['target', 'llm_reponse']]
results_targets['llm_reponse'] = results_targets['llm_reponse'].astype(int)
results_targets['target'] = results_targets['target'].astype(int)

#%%
conf_matrix = confusion_matrix(results_targets['target'], results_targets['llm_response'], labels=[i for i in range(10)])

print('--- results of the confusion matrix ----')
print(conf_matrix)

