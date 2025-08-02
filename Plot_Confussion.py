import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
no_of_dataset = 2


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
