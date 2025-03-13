import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data = pd.read_csv('prediction_results_test.csv', index_col=0)
print(data.cancer.unique())
print('Pearson correlation for all, train set:', pearsonr(data.true_ic50, data.pred_ic50))
corr = pearsonr(data.true_ic50, data.pred_ic50)[0]
plt.scatter(data.true_ic50, data.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Model IC50')
plt.title('Predicted vs true IC50 for all drugs and cell lines')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.savefig('ic50_train.png')
plt.cla()

# ovarian cancer
data_subset = data[data.cancer=='OV']
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
print('Pearson correlation for OV, train_set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.xlabel('True IC50')
plt.ylabel('Model IC50')
plt.title('Predicted vs true IC50 for ovarian cancer, all drugs')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.savefig('ic50_train_ov.png')
plt.cla()

# pancreatic cancer
data_subset = data[data.cancer=='PAAD']
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
print('Pearson correlation for PAAD, train set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.xlabel('True IC50')
plt.ylabel('Model IC50')
plt.title('Predicted vs true IC50 for pancreatic cancer, all drugs')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.savefig('ic50_train_pd.png')
plt.cla()

# pancreatic cancer test
data = pd.read_csv('prediction_results_test_PAAD.csv', index_col=0)
data_subset = data[data.cancer=='PAAD']
print('Pearson correlation for PAAD, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 for pancreatic cancer, all drugs')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.savefig('ic50_test_PAAD.png')
plt.cla()

# OV cancer test
data = pd.read_csv('prediction_results_test_OV.csv', index_col=0)
data_subset = data[data.cancer=='OV']
print('Pearson correlation for OV, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 for ovarian cancer, all drugs')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.savefig('ic50_test_OV.png')
plt.cla()

# all cancers test
data = pd.read_csv('prediction_results_test_with_mut_with_gexp_without_methy_256_256_256_bn_relu_GAP.csv', index_col=0)
data_subset = data
print('Pearson correlation for all cancers, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 for all cancers, all drugs')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.savefig('ic50_test_5.png')
plt.cla()
