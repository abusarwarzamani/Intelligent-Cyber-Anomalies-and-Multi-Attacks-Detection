import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_ENSEMBLE import Model_Ensemble


def objfun_feat(Soln):
    Feat = Global_Vars.Feat

    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            feat = Feat[:, :7] * sol
            varience = np.var(feat)
            corr_coef = np.corrcoef(feat)
            Corr = np.mean(corr_coef)
            Fitn[i] = 1 / (Corr + varience)  # corrcoef + varience
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        feat = Feat[:, :7] * sol
        varience = np.var(feat)
        corr_coef = np.corrcoef(feat)
        Corr = np.mean(corr_coef)
        Fitn = 1 / (Corr + varience)  # corrcoef + varience
    return Fitn

def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1/(Eval[4] + Eval[7])
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / (Eval[4] + Eval[7])
        return Fitn
