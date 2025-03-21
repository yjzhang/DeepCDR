#get drug features using Deepchem library
import os
import deepchem as dc
from rdkit import Chem
import numpy as np
import pandas as pd
import hickle as hkl


drug_smiles_file = '../data/223drugs_pubchem_smiles.txt'
save_dir = '../data/GDSC/drug_graph_feat'
pubchemid2smile = {item.split('\t')[0]:item.split('\t')[1].strip() for item in open(drug_smiles_file).readlines()}
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
molecules = []
for each in pubchemid2smile.keys():
	print(each)
	molecules=[]
	molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
	featurizer = dc.feat.graph_features.ConvMolFeaturizer()
	mol_object = featurizer.featurize(datapoints=molecules)
	features = mol_object[0].atom_features
	degree_list = mol_object[0].deg_list
	adj_list = mol_object[0].canon_adj_list
	hkl.dump([features,adj_list,degree_list],'%s/%s.hkl'%(save_dir,each))

drug_smiles_file = '../data/cancer_drugs_smiles.tsv'
save_dir = '../data/new_drugs/drug_graph_feat'
smiles_data = pd.read_csv(drug_smiles_file, sep='\t')
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
molecules = []
for _, row in smiles_data.iterrows():
	smiles = row.smiles
	print(row.cid)
	molecules=[]
	molecules.append(Chem.MolFromSmiles(smiles))
	featurizer = dc.feat.graph_features.ConvMolFeaturizer()
	mol_object = featurizer.featurize(datapoints=molecules)
	features = mol_object[0].atom_features
	degree_list = mol_object[0].deg_list
	adj_list = mol_object[0].canon_adj_list
	hkl.dump([features, adj_list,degree_list],'%s/%s.hkl'%(save_dir, row.cid))

# convert beatAML
drug_smiles_file = '../data/beataml_smiles.csv'
save_dir = '../data/beataml/drug_graph_feat'
smiles_data = pd.read_csv(drug_smiles_file)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
molecules = []
for _, row in smiles_data.iterrows():
	smiles = row.canonical_smiles
	print(row.cid)
	molecules=[]
	molecules.append(Chem.MolFromSmiles(smiles))
	featurizer = dc.feat.graph_features.ConvMolFeaturizer()
	mol_object = featurizer.featurize(datapoints=molecules)
	features = mol_object[0].atom_features
	degree_list = mol_object[0].deg_list
	adj_list = mol_object[0].canon_adj_list
	hkl.dump([features, adj_list,degree_list],'%s/%s.hkl'%(save_dir, row.cid))
