import numpy as np
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(description="Match gene name and order for RNA and protein data")

parser.add_argument('--pro_dir', type=str, default='', help='the protein abundance data file path')
parser.add_argument('--rna_dir', type=str, default='', help='the RNA expression data file path')
parser.add_argument('--out_dir', type=str, default='./results/matched/', help='the output directory to save processed data')
args = parser.parse_args()
# Check if the output directory exists, if not, create it
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
pro_list = pd.read_csv('./data/pro_list.csv')
pro_list = pro_list['pro'].tolist()
rna_list = pd.read_csv('./data/gene_list.csv')
rna_list = rna_list['gene'].tolist()

# Load the data
if args.pro_dir == '' and args.rna_dir == '':
    print('Please provide at least one data path to process')
    exit(0)
if args.pro_dir != '':
    if not os.path.exists(args.pro_dir):
        print('The protein abundance data file path does not exist')
        exit(0)
    else:
        pro_dir = args.pro_dir
        data_name = pro_dir.split('/')[-1]
        new_dir = data_name.split('.')[0] + '_matched.csv'
        one_pro_df = pd.read_csv(pro_dir)
        diff_list = list(set(pro_list).difference(set(one_pro_df.columns.tolist())))
        n_pro = one_pro_df.shape[1]-1
        n_ov = n_pro-len(diff_list)
        one_pro_df[diff_list] = 0
        new_pro_df = one_pro_df[['PID']+pro_list]
        print('The number of proteins in data:', n_pro)
        print("The number of proteins shared with T2Pdecoder's 5738 protien:", n_ov)
        print('The number of proteins have been imputed', len(diff_list))
        new_pro_df.to_csv(args.out_dir+'/'+new_dir, index=False)
        print('Has done the protein abundance data processing!')

if args.rna_dir != '':
    if not os.path.exists(args.rna_dir):
        print('The RNA expression data file path does not exist')
        exit(0)
    else:
        rna_dir = args.rna_dir
        data_name = rna_dir.split('/')[-1]
        new_dir = data_name.split('.')[0] + '_matched.csv'
        one_rna_df = pd.read_csv(rna_dir)
        diff_list = list(set(rna_list).difference(set(one_rna_df.columns.tolist())))
        n_rna = one_rna_df.shape[1]-1
        n_ov = n_rna-len(diff_list)
        one_rna_df[diff_list] = 0
        new_rna_df = one_rna_df[['PID']+rna_list]
        print('The number of genes in data:', n_rna)
        print("The number of genes shared with T2Pdecoder's 18860 genes:", n_ov)
        print('The number of genes have been imputed', len(diff_list))
        new_rna_df.to_csv(args.out_dir+'/'+new_dir, index=False)
        print('Has done the RNA expression data processing!')
