#!/usr/bin/env python

import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams["font.size"] = 15

data = pd.read_csv('prediction_results_test.csv', index_col=0)
print(data.cancer.unique())
print('Pearson correlation for all, train set:', pearsonr(data.true_ic50, data.pred_ic50))
corr = pearsonr(data.true_ic50, data.pred_ic50)[0]
plt.scatter(data.true_ic50, data.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Model IC50')
plt.title('Predicted vs true IC50 for all drugs and cell lines')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.tight_layout()
plt.savefig('ic50_train.png', dpi=300)
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
plt.tight_layout()
plt.savefig('ic50_train_ov.png', dpi=300)
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
plt.tight_layout()
plt.savefig('ic50_train_pd.png', dpi=300)
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
plt.tight_layout()
plt.savefig('ic50_test_PAAD.png', dpi=300)
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
plt.tight_layout()
plt.savefig('ic50_test_OV.png', dpi=300)
plt.cla()

# all cancers test
data = pd.read_csv('prediction_results_test_with_mut_with_gexp_without_methy_256_256_256_bn_relu_GAP.csv', index_col=0)
data_subset = data
print('Pearson correlation for all cancers, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.tight_layout()
plt.savefig('ic50_test_5.png', dpi=300)
plt.cla()

# gexp only
data = pd.read_csv('prediction_results_test_without_mut_with_gexp_without_methy_256_256_256_bn_relu_GAP.csv', index_col=0)
data_subset = data
print('Pearson correlation for all cancers, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 - Gene Expression only')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.tight_layout()
plt.savefig('ic50_test_gexp_only.png', dpi=300)
plt.cla()

# mut only
data = pd.read_csv('prediction_results_test_with_mut_without_gexp_without_methy_256_256_256_bn_relu_GAP.csv', index_col=0)
data_subset = data
print('Pearson correlation for all cancers, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 - Mutation only')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.tight_layout()
plt.savefig('ic50_test_mut_only.png', dpi=300)
plt.cla()

# methy only
data = pd.read_csv('prediction_results_test_without_mut_without_gexp_with_methy_256_256_256_bn_relu_GAP.csv', index_col=0)
data_subset = data
print('Pearson correlation for all cancers, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 - Methylation only')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.tight_layout()
plt.savefig('ic50_test_methy_only.png', dpi=300)
plt.cla()

# gexp only
data = pd.read_csv('prediction_results_beataml_test.csv', index_col=0)
data_subset = data
print('Pearson correlation for BeatAML, test set:', pearsonr(data_subset.true_ic50, data_subset.pred_ic50))
corr = pearsonr(data_subset.true_ic50, data_subset.pred_ic50)[0]
plt.scatter(data_subset.true_ic50, data_subset.pred_ic50, s=1)
plt.xlabel('True IC50')
plt.ylabel('Predicted IC50')
plt.title('Predicted vs true IC50 - BeatAML, Gene Expression only')
plt.text(5, -4, 'Pearson R: ' + str(round(corr, 2)))
plt.tight_layout()
plt.savefig('ic50_test_beataml.png', dpi=300)
plt.cla()

