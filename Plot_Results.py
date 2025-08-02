import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
no_of_dataset = 2


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])

def plot_Con_results():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA', ' BCO', 'DA', 'OOA', 'RUP-OOA']
    Classifier = ['TERMS', 'LSTM', 'DTCNN', 'RNN', 'DA-RNN', 'DMRF+DA-RNN']

    for i in range(Fitness.shape[0]):
        Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ Configuration', i + 1, 'Statistical Report ',
              '------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[i]

        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=2, marker='*', markerfacecolor='red',
                 markersize=5, label='AOA')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=2, marker='*', markerfacecolor='green',
                 markersize=5, label='BCO')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=2, marker='*', markerfacecolor='blue',
                 markersize=5, label='DA')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=2, marker='*', markerfacecolor='magenta',
                 markersize=5, label='OOA')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=2, marker='*', markerfacecolor='black',
                 markersize=5, label='RUP-OOA')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%d.png" % (i + 1))
        plt.show()


def PLot_ROC():
    lw = 2

    cls = ['LSTM', 'DTCNN', 'RNN', 'DA-RNN', 'DMRF+DA-RNN']
    for a in range(no_of_dataset):  # For Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "crimson", "gold", "lime", "black"])  # "cornflowerblue","darkorange", "aqua"
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_1_ROC_%s_.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def Plot_Confusion():
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    for n in range(no_of_dataset):
        ax = plt.subplot()
        value = eval1[n, 3, 4, :4]
        tp = value[0]
        tn = value[1]
        fp = value[2]
        fn = value[3]
        act = [tp, fp]
        pred = [fn, tn]
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        cm = [act, pred]
        # cm = confusion_matrix(np.asarray(Actual[n]).argmax(axis=1), np.asarray(Predict[n]).argmax(axis=1))
        # cm = confusion_matrix(np.asarray(Actual[n]), np.asarray(Predict[n]))
        # Table = PrettyTable()
        # for i in range(len(cm[:, 0])):
        #     Table.add_row(cm[i, :])
        # print('-------------------------------------------------- confusion - Dataset', n + 1,
        #       '--------------------------------------------------')
        # print(Table)
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        plt.title('Accuracy')
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()


def plot_results__():  # For classification
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ["accuracy", "sensitivity", "specificity", "precision", "FPR", "FNR", "NPV", "FDR", "F1_score", "MCC"]
    Graph_Terms = [0, 1, 2, 3]
    Algorithm = ['TERMS', 'CSO', ' EOO', 'LOA', 'OOA', 'PROPOSED']
    Classifier = ['TERMS', 'CNN', 'DMRF', 'RNN', 'Ensemble', 'PROPOSED']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 3, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Epoch 250 - Dataset', i + 1, 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Epoch 250 - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [1, 2, 3, 4, 5]
    for i in range(eval1.shape[0]):
        Graph = np.zeros((len(Graph_Terms), eval1.shape[1], eval1.shape[2]))
        for j in range(len(Graph_Terms)):
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[j, k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[j, k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

        for p in range(len(Graph_Terms)):
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[p, 0, :5], color='b', width=0.10, label="50 EPOCH")
            ax.bar(X + 0.10, Graph[p, 1, :5], color='aqua', width=0.10, label="100 EPOCH")
            ax.bar(X + 0.20, Graph[p, 2, :5], color='r', width=0.10, label="150 EPOCH")
            ax.bar(X + 0.30, Graph[p, 3, :5], color='lime', width=0.10, label="200 Epoch")
            ax.bar(X + 0.40, Graph[p, 4, :5], color='k', width=0.10, label="250 EPOCH")

            plt.xticks(X + 0.10, ('CSO', ' EOO', 'LOA', 'OOA', 'PROPOSED'), rotation=8)
            plt.ylabel(Terms[p])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_comp_%s_%s_bar_Alg.png" % (i + 1, p + 1, Terms[p])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.11, 0.11, 0.75, 0.75])
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[p, 0, 5:10], color='r', width=0.10, label="50 EPOCH")
            ax.bar(X + 0.10, Graph[p, 1, 5:10], color='lime', width=0.10, label="100 EPOCH")
            ax.bar(X + 0.20, Graph[p, 2, 5:10], color='b', width=0.10, label="150 EPOCH")
            ax.bar(X + 0.30, Graph[p, 3, 5:10], color='m', width=0.10, label="200 Epoch")
            ax.bar(X + 0.40, Graph[p, 4, 5:10], color='k', width=0.10, label="250 EPOCH")

            plt.xticks(X + 0.10, ('CNN', 'DMRF', 'RNN', 'Ensemble', 'PROPOSED'), rotation=10)
            plt.ylabel(Terms[p])
            plt.tight_layout()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_comp_%s_%s_bar_net.png" % (i + 1, p + 1, Terms[p])
            plt.savefig(path1)
            plt.show()


### For learning percentage
def plot_ACTIVATION():
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 6, 8, 9]
    Algorithm = ['TERMS', 'AOA', ' BCO', 'DA', 'OOA', 'RUP-OOA']
    Classifier = ['TERMS', 'LSTM', 'DTCNN', 'RNN', 'DA-RNN', 'DMRF+DA-RNN']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Terms)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value1[j, :])
        # print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Algorithm Comparison',
        #       '--------------------------------------------------')
        # print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2] + 1))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 5:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            # plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
            #          label="AOA")
            # plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
            #          label="BCO")
            # plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
            #          label="DA")
            # plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
            #          label="OOA")
            # plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='white', markersize=12,
            #          label="RUP-OOA")
            # # # plt.plot(learnper, Graph[:, 5], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
            # # #          label="GRSO-CRN")
            # plt.xticks(learnper, ('Linear', 'Relu', 'Tanh', 'Sigmaoid', 'Softmax', 'Leaky_Relu'))
            # plt.xlabel('Activation Function')
            # plt.ylabel(Terms[Graph_Terms[j]])
            # plt.legend(loc=4)
            # path1 = "./Results/Dataset_%s_Activation_%s_line_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            # plt.savefig(path1)
            # plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            Algorithm = ['TERMS', 'AOA', ' BCO', 'DA', 'OOA', 'RUP-OOA']
            Classifier = ['TERMS', 'LSTM', 'DTCNN', 'RNN', 'DA-RNN', 'DMRF+DA-RNN']
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="DTCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="RNN")
            ax.bar(X + 0.30, Graph[:, 8], color='c', width=0.10, label="DA-RNN")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="DMRF+DA-RNN")
            # ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="PROPOSED")
            plt.xticks(X + 0.10, ('Linear', 'Relu', 'Tanh', 'Sigmaoid', 'Softmax', 'Leaky_Relu'))
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_Activation_%s_bar_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()



def plot_results_Feat():
    # matplotlib.use('TkAgg')
    eval1 = np.load('Eval_all__.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 6, 8]
    Algorithm = ['TERMS', 'AOA', ' BCO', 'DA', 'OOA', 'RUP-OOA']
    Classifier = ['TERMS', 'LSTM', 'DTCNN', 'RNN', 'DA-RNN', 'DMRF+DA-RNN']
    Feature = ['Autoencoder Feature', 'RBM Feature', 'Combined Feature', 'Weighted Feature']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]
    #
    #     Table = PrettyTable()
    #     Table.add_column(Algorithm[0], Terms)
    #     for j in range(len(Algorithm) - 1):
    #         Table.add_column(Algorithm[j + 1], value1[j, :])
    #     print('-------------------------------------------------- Dataset ', 1, 'Algorithm Comparison',
    #           '--------------------------------------------------')
    #     print(Table)
    #
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- ' + Feature[i] + ' Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    # learnper = [35, 45, 55, 65, 75, 85]
    # for j in range(len(Graph_Terms)):
    #     Graph = np.zeros((eval1.shape[0], eval1.shape[2] + 1))
    #     for k in range(eval1.shape[0]):
    #         for l in range(eval1.shape[2]):
    #             if j == 9:
    #                 Graph[k, l] = eval1[k, 4, l, Graph_Terms[j] + 4]
    #             else:
    #                 Graph[k, l] = eval1[k, 4, l, Graph_Terms[j] + 4]

    Feature = ['Autoencoder Feature', 'RBM Feature', 'Combined Feature', 'Weighted Feature']
    learnper = [1, 2, 3, 4, 5]
    # for i in range(eval1.shape[0]):
    Graph = np.zeros((len(Graph_Terms), eval1.shape[0], eval1.shape[2]))
    for j in range(len(Graph_Terms)):
        for k in range(eval1.shape[0]):
            for l in range(eval1.shape[2]):
                if j == 9:
                    Graph[j, k, l] = eval1[k, 1, l, Graph_Terms[j] + 4]
                else:
                    Graph[j, k, l] = eval1[k, 1, l, Graph_Terms[j] + 4]

    for p in range(len(Graph_Terms)):
        # plt.plot(learnper, Graph[p, 0, 0:5], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
        #          label="RBM Feature")
        # plt.plot(learnper, Graph[p, 1, 0:5], color='g', linewidth=3, marker='o', markerfacecolor='m', markersize=12,
        #          label="Autoencoder Feature")
        # plt.plot(learnper, Graph[p, 2, 0:5], color='k', linewidth=3, marker='o', markerfacecolor='lime', markersize=12,
        #          label="DEEP Feature")
        # # plt.plot(learnper, Graph[p, 3, 0:5], color='m', linewidth=3, marker='o', markerfacecolor='yellow',
        # #          markersize=12,
        # #          label="EEG Feature")
        # # plt.plot(learnper, Graph[p, 4, 0:5], color='k', linewidth=3, marker='o', markerfacecolor='magenta',
        # #          markersize=12,
        # #          label="Fused Feature")
        #
        # plt.xticks(learnper, (['EOO-AHDNet', ' DTBO-AHDNet', 'COA-AHDNet', 'BOS-AHDNet', 'UNU-BOSA-AHDNet']), rotation=15)
        # # plt.axis('on')
        # # plt.grid(False)
        # # plt.xlabel('No of Dataset')
        # plt.ylabel(Terms[p])
        # plt.legend(loc=4)
        # path1 = "./Results/Features_%s_line.png" % (Terms[p])
        # plt.savefig(path1)
        # plt.show()

        fig = plt.figure()
        # temp = Graph[p, :, 4]
        # Graph[p, :, 9] = temp
        ax = fig.add_axes([0.125, 0.125, 0.75, 0.75])
        X = np.arange(5)
        Feature = ['Autoencoder Feature', 'RBM Feature', 'Combined Feature', 'Weighted Feature']
        ax.bar(X + 0.00, Graph[p, 0, 5:10], color='r', edgecolor='black', width=0.10, label="Autoencoder Feature")
        ax.bar(X + 0.10, Graph[p, 1, 5:10], color='lime', edgecolor='black', width=0.10, label="RBM Feature")
        ax.bar(X + 0.20, Graph[p, 2, 5:10], color='b', edgecolor='black', width=0.10, label="Combined Feature")
        ax.bar(X + 0.30, Graph[p, 3, 5:10], color='k', edgecolor='black', width=0.10, label="Weighted Feature")
        # ax.bar(X + 0.30, Graph[p, 3, 5:10], color='m', edgecolor='black', width=0.10, label="EEG Feature")
        # ax.bar(X + 0.40, Graph[p, 4, 5:10], color='k', edgecolor='black', width=0.10, label="Fused Feature")
        # ax.bar(X + 0.50, Graph[p, 5, 5:10], color='k', edgecolor='black', width=0.10, label="85 %")

        plt.xticks(X + 0.20, ('LSTM', 'DTCNN', 'RNN', 'DA-RNN', 'DMRF+DA-RNN'))
        # plt.xlabel('Learning %')
        plt.ylabel(Terms[p])
        # plt.legend(loc=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        plt.ylim([70, 100])
        path1 = "./Results/Features_%s_bar.png" % (Terms[p])
        plt.savefig(path1)
        plt.show()




def PLOT_RESULTS():
    # plot_Con_results()
    # PLot_ROC()
    # Plot_Confusion()
    # plot_ACTIVATION()
    plot_results_Feat()


if __name__ == '__main__':
    PLOT_RESULTS()
