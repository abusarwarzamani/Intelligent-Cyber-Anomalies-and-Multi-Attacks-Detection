import numpy as np
import pandas as pd
import os
from numpy import matlib
import random as rn
from Global_Vars import Global_Vars
from Model_Autoencoder import Model_AutoEn
from Model_CNN import Model_CNN
from Model_DMRF import Model_DMRF
from Model_ENSEMBLE import Model_Ensemble
from Model_RBM import Model_RBM
from Model_RNN import Model_RNN
from Objective_Function import objfun_cls, objfun_feat
from PROPOSED import PROPOSED, OOA, LOA, EOO, CSO
from Plot_Results import PLOT_RESULTS


no_of_dataset = 2

#  Read the dataset
an = 0
if an == 1:
    Dataset = './Dataset/Dataset_1/APA-DDoS-Dataset/APA-DDoS-Dataset.csv'
    Data = pd.read_csv(Dataset)
    Data.drop('frame.time', inplace=True, axis=1)
    data_1 = np.asarray(Data)
    data1 = data_1[:, :-1]

    tar = data_1[:, -1]
    # tar = pd.read_csv(File, usecols=['churn'])
    Uni = np.unique(tar)
    uni = np.asarray(Uni)
    Target_1 = np.zeros((tar.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(tar == uni[j])
        Target_1[ind, j] = 1
    np.save('Data_1.npy', data1)  # Save the Dataset_1
    np.save('Target_1.npy', Target_1)  # Save the Target_1

# Read the dataset 2
an = 0
if an == 1:
    Dataset = './Dataset/Dataset_2/wustl-scada-2018.csv'  # Path of the dataset_5
    Data = pd.read_csv(Dataset)
    data_1 = np.asarray(Data)
    data1 = data_1[:, :-1][500000:700000, :]
    tar = data_1[:, -1][500000:700000]
    # tar = pd.read_csv(File, usecols=['churn'])
    Uni = np.unique(tar)
    uni = np.asarray(Uni)
    Target_1 = np.zeros((tar.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(tar == uni[j])
        Target_1[ind, j] = 1
    np.save('Data_2.npy', data1)  # Save the Dataset_1
    np.save('Target_2.npy', Target_1)  # Save the Target_1

# deep feature extraction
# RBM feature
an = 0
if an == 1:
    for n in range(no_of_dataset):
        print(n)
        Features = []
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)[:10000, :]
        if n == 0:
            RBM = Model_RBM((Data[:, 2:]).astype('float'))
        else:
            RBM = Model_RBM((Data).astype('float'))
        np.save('Feat_1_' + str(n + 1) + '.npy', RBM)

# AutoEncoder feature
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Features = []
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        if n == 0:
            AutoEncoder = Model_AutoEn(((Data[:, 2:]).astype('float')), target)
        else:
            AutoEncoder = Model_AutoEn(((Data).astype('float')), target)
        np.save('Feat_2_' + str(n + 1) + '.npy', AutoEncoder)

# Optimization for Feature Selection And Weight Optimization
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feat_1_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feat_2 = np.load('Feat_2_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        Global_Vars.Feat_1 = Feat_1.astype(np.float)
        Global_Vars.Feat_2 = Feat_2.astype(np.float)
        Npop = 10
        Chlen = 24  # (12 + 12) for feature selection
        xmin = matlib.repmat((0.01 * np.ones(24)), Npop, 1)
        xmax = matlib.repmat((0.99 * np.ones(24)), Npop, 1)
        fname = objfun_feat
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("CSO...")
        [bestfit1, fitness1, bestsol1, time1] = CSO(initsol, fname, xmin, xmax, Max_iter)  # CSO

        print("EOO..")
        [bestfit2, fitness2, bestsol2, time2] = EOO(initsol, fname, xmin, xmax, Max_iter)  # EOO

        print("LOA...")
        [bestfit4, fitness4, bestsol4, time3] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("OOA...")
        [bestfit3, fitness3, bestsol3, time4] = OOA(initsol, fname, xmin, xmax, Max_iter)  # OOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_feat = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        # BestSol_feat = [initsol[0,:], initsol[1,:], initsol[2,:], initsol[3,:], initsol[4,:]]
        np.save('BestSol_FEAT_' + str(n + 1) + '.npy', BestSol_feat)  # Save the BestSol_FEAT


# ## feature Concatenation for Feature Fusion
an = 0
if an == 1:
    Feat_1 = np.load('Feat_1.npy', allow_pickle=True)  # Load the Feat 1
    Feat_2 = np.load('Feat_2.npy', allow_pickle=True)  # Load the Feat 1
    bests = np.load('BestSol_FEAT.npy', allow_pickle=True)  # Load the Bestsol Feat
    weighted_feature_1 = Feat_1 * bests[4, :12]
    weighted_feature_2 = Feat_2[:, :12] * bests[4, 12:]
    Feature = np.concatenate(([weighted_feature_1, weighted_feature_2]), axis=1)
    np.save('Feature.npy', Feature)  # Save the Feature

# optimization for Classification
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 4  #
        xmin = matlib.repmat([50, 0, 50, 5], Npop, 1)
        xmax = matlib.repmat([100, 3, 100, 255], Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("EBOA...")
        [bestfit1, fitness1, bestsol1, time1] = CSO(initsol, fname, xmin, xmax, Max_iter)  # EBOA

        print("GMO...")
        [bestfit2, fitness2, bestsol2, time2] = EOO(initsol, fname, xmin, xmax, Max_iter)  # GMO

        print("COA...")
        [bestfit4, fitness4, bestsol4, time3] = LOA(initsol, fname, xmin, xmax, Max_iter)  # COA

        print("DTBO...")
        [bestfit3, fitness3, bestsol3, time4] = OOA(initsol, fname, xmin, xmax, Max_iter)  # DTBO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved DTBO
        BestSol_CLS = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        # BestSol_CLS = [initsol[0, :], initsol[1, :], initsol[2, :], initsol[3, :], initsol[4, :]]
        np.save('BestSol_CLS_' + str(n + 1) + '.npy', BestSol_CLS)  # Bestsol classification

# classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feature = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)[:Feature.shape[0]]  # Load the Target
        BestSol = np.load('BestSol_CLS_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Bestsol Classification
        Feat = Feature
        EVAL = []
        Batchsize = [4, 16, 32, 64, 128]
        for Batch in range(len(Batchsize)):
            Batchsizec = round(Feat.shape[0] * Batchsize[Batch])
            Train_Data = Feat[:Batchsizec, :]
            Train_Target = Target[:Batchsizec, :]
            Test_Data = Feat[Batchsizec:, :]
            Test_Target = Target[Batchsizec:, :]
            Eval = np.zeros((10, 14))
            for j in range(BestSol.shape[0]):
                print(Batch, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred0 = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred2 = Model_DMRF(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred3 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred4 = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, [0, 0, 50, 0.01])
            Eval[9, :], pred5 = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
        np.save('Eval_all.npy', Eval_all)  # Save the Eval all


PLOT_RESULTS()