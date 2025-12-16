from LayerRNNModel import LayerRNNModel
import matplotlib.pyplot as plt
from GBTModel import GBTModel

def cross_fold_verify(k=3):

    accuracy = []

    for idx in range(k):
        model = GBTModel(f'models/{idx}', random_state=idx)

        acc = model.train_formation(n_estimators=125)
        accuracy.append(acc)

    return accuracy


def graph_n_estimators():
    accuracy = []
    f1_total = []

    model = GBTModel('models/0')

    for n_estimators in range(25, 225, 25):
        f1, acc = model.train_formation(n_estimators=n_estimators)

        accuracy.append(acc)
        f1_total.append(f1)

    plt.plot(range(25, 225, 25), accuracy)

    plt.xlabel("Number of Estimators")
    plt.ylabel('Accuracy')

    plt.grid(True)
    plt.savefig('form_acc_plot.png')

    plt.close()

    plt.plot(range(25, 225, 25), f1_total)

    plt.xlabel("Number of Estimators")
    plt.ylabel('Macro F1-Score')

    plt.grid(True)
    plt.savefig('form_f1_plot.png')

def graph_learning_rates():
    lrs = [1e-2, 1e-3, 1e-4, 1e-5]

    model = LayerRNNModel('models/test1/test1')

    train_losses = []
    test_losses = []

    for lr in lrs:
        train, test = model.train(retrain=False, lr=lr, max_epochs=5)
        train_losses.append(train)
        test_losses.append(test)

    epochs = range(1, len(train_losses[0]) + 1)

    for idx in range(len(train_losses)):
        plt.plot(epochs, train_losses[idx], label=f'Train Loss {lrs[idx]}')
        plt.plot(epochs, test_losses[idx], label=f'Test Loss {lrs[idx]}')

    plt.xlabel("Epochs")
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')