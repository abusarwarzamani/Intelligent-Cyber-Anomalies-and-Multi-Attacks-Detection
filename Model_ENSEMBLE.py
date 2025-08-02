from Evaluation import evaluation
from Model_DMRF import Model_DMRF
from Model_RNN import Model_RNN


def Model_Ensemble(train_data, train_target, test_data, test_target, sol):
    Eval, pred_LSTM = Model_DMRF(train_data, train_target, test_data, test_target, sol[1:])
    Eval, Pred_DNN = Model_RNN(train_data, train_target, test_data, test_target, sol[0])

    pred = (pred_LSTM + Pred_DNN) / 2

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval
