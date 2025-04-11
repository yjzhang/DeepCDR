import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
from keras.utils import multi_gpu_model,plot_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from model import KerasMultiSourceGCNModel
import hickle as hkl
import scipy.sparse as sp
import argparse

# TODO: rewrite this to only do evaluation.

####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('-no_use_mut', dest='use_mut', action='store_false', default=True, help='use gene mutation or not')
parser.add_argument('-no_use_gexp', dest='use_gexp', action='store_false', default=True, help='use gene expression or not')
parser.add_argument('-no_use_methy', dest='use_methy', action='store_false', default=True, help='use methylation or not')

# test cancer (a cancer type to not use?)
parser.add_argument('-test_cancer', dest='test_cancer', type=str, default='', help='Is there a cancer type to exclude from the training data?')


parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256,256,256],help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut, use_gexp, use_methy = args.use_mut, args.use_gexp, args.use_methy
test_cancer = args.test_cancer
israndom=args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut')+'_'+('with_gexp' if use_gexp else 'without_gexp')+'_'+('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str, args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')

model_suffix = model_suffix + '_' +GCN_deploy

if test_cancer:
    model_suffix += '_' + test_cancer

model_suffix = 'without_mut_with_gexp_without_methy_256_256_256_bn_relu_GAP_beataml'

print(model_suffix)

####################################Constants Settings###########################
DPATH = '../data'

# TODO: evaluate for BeatAML

Drug_info_file = '%s/beataml_smiles.csv'%DPATH

Drug_feature_file = '%s/beataml/drug_graph_feat'%DPATH
#Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv'%DPATH
Cancer_response_exp_file = '%s/beataml/drug_response.csv'%DPATH
Gene_expression_file = '%s/beataml/gene_exp.csv'%DPATH
#Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'%DPATH
Max_atoms = 100


def generate_test_data(Drug_info_file, Drug_feature_file, Gene_expression_file):
    """
    """
    # TODO: generate data from beatAML data
    #drug_id --> pubchem_id

    #load demap cell lines genomic mutation features
    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        filename = '%s/%s'%(Drug_feature_file, each)
        print('drug filename:', filename)
        feat_mat, adj_list, degree_list = hkl.load('%s/%s'%(Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    
    # data_idx: cell line name, pubchem id for drug, cancer type
    data_idx = []
    for pubchem_id in drug_pubchem_id_set:
        for each_cellline in gexpr_feature.index:
            data_idx.append((each_cellline, pubchem_id, )) 
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx), nb_celllines, nb_drugs))
    return drug_feature, gexpr_feature, data_idx


def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix


def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms, feat_mat.shape[-1]), dtype='float32')
    adj_mat = np.zeros((Max_atoms, Max_atoms), dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms, feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0], :] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    adj_ = adj_mat[:len(adj_list), :len(adj_list)]
    adj_2 = adj_mat[len(adj_list):, len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2    
    return [feat, adj_mat]


def FeatureExtract(data_idx, drug_feature, gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    gexpr_data = np.zeros((nb_instance, nb_gexpr_features), dtype='float32') 
    target = np.zeros(nb_instance, dtype='float32')
    for idx in range(nb_instance):
        cell_line_id, pubchem_id = data_idx[idx]
        #modify
        feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
        #fill drug data, padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list)
        #randomlize X A
        gexpr_data[idx, :] = gexpr_feature.loc[cell_line_id].values
        cancer_type_list.append([cell_line_id, pubchem_id])
    return drug_data, gexpr_data, target, cancer_type_list
    

def ModelEvaluate(model, X_drug_data_test, X_gexpr_data_test, Y_test):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    X = [X_drug_feat_data_test, X_drug_adj_data_test]
    if use_gexp:
        X.append(X_gexpr_data_test)
    Y_pred = model.predict(X)
    overall_pcc = pearsonr(Y_pred[:, 0], Y_test)[0]
    print("The overall Pearson's correlation is %.4f."%overall_pcc)
    return Y_pred
    

if __name__=='__main__':
    # TODO: 1. load model
    from layers.graph import GraphLayer, GraphConv

    print('Loading model...')
    model = load_model('../checkpoint/MyBestDeepCDR_{0}.h5'.format(model_suffix),
            custom_objects={'GraphLayer': GraphLayer,
                'GraphConv': GraphConv})

    Drug_feature_file = '../data/beataml/drug_graph_feat' 
    drug_feature, gexpr_feature, data_idx = generate_test_data(Drug_info_file, Drug_feature_file, Gene_expression_file)

    print('Number of test points:', len(data_idx))

    #Extract features for training and test 
    X_drug_data_test, X_gexpr_data_test, Y_test, cancer_type_test_list = FeatureExtract(data_idx, drug_feature, gexpr_feature) 

    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  
    
    validation_data = [[X_drug_feat_data_test, X_drug_adj_data_test, X_gexpr_data_test], Y_test]

    print('Evaluating model...')
    Y_pred = ModelEvaluate(model, X_drug_data_test, X_gexpr_data_test, Y_test)


    # these are the results of testing on the same dataset as the training set...
    data = [x+tuple(y) for x, y in zip(data_idx, Y_pred)]
    df = pd.DataFrame(data)
    df.columns = ['cell_line', 'pubchem_id', 'pred_ic50']
    df.to_csv('prediction_results_beataml_tes.csv')

