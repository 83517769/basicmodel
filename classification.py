import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm
from torch.optim import Adam
from torchvision.transforms import ToTensor
import statistics as stats
import pandas as pd
import vgg16
from pathlib import Path
import numpy as np

seed =1
np.random.seed(seed)
torch.cuda.manual_seed(seed)

def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    percent_correct = total_correct / (total_correct + total_incorrect)
    return precision, percent_correct


columns = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'total']
columns = np.array(columns)
columns = columns.reshape(1, -1)
dt = pd.DataFrame(columns)
dt.to_csv("precis_data2.csv", mode='a', index=False, header=None)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    num_classes = 10
    fully_supervised = False
    reload = 1229
    run_id = 6
    epochs = 150
    LR = 1e-3
    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    # ds = datasets.MNIST(root=r'c:\data\tv', transform=ToTensor(), download=True)
    ds = CIFAR10(r'c:\data\tv', download=True, transform=ToTensor())
    len_train = len(ds) // 10 * 9
    len_test = len(ds) - len_train
    train, test = random_split(ds, [len_train, len_test])
    train_l = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_l = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)

    # if fully_supervised:
    #     classifier = nn.Sequential(
    #         models.Encoder(),
    #         models.Classifier()
    #     ).to(device)
    # else:
    #     classifier = models.DeepInfoAsLatent('run5', '1990').to(device)
    #     if reload is not None:
    #         classifier = torch.load(f'c:/data/deepinfomax/models_MNIST/run{run_id}/w_dim{reload}.mdl')
    classifier = models.DeepInfoAsLatent('run5', '1990').to(device)
    # classifier = torch.load(f'c:/data/deepinfomax/models/run{run_id}/w_dim{reload}.mdl')
    optim = torch.optim.SGD(classifier.parameters(), lr=LR, momentum=0.8, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    print(classifier)

    for epoch in range(1,   epochs):
        if epoch % 40 == 0:
            LR = LR*0.1
            print("Learning rate decay to :{}".format(LR))
        ll = []
        batch = tqdm(train_l, total=len_train // batch_size)
        for x, target in batch:
            #print(x.size())
            x = x.to(device)
            # x = x.view(256, 28*28*1)
            #print(x.size())
            target = target.to(device)
            optim.zero_grad()
            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(f'{epoch} Train Loss: {stats.mean(ll)}')
            loss.backward()
            optim.step()

        confusion = torch.zeros(num_classes, num_classes)
        batch = tqdm(test_l, total=len_test // batch_size)
        ll = []
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)
            # x = x.view(256, 28 * 28 * 1)
            y = classifier(x)
            loss = criterion(y, target)
            ll.append(loss.detach().item())
            batch.set_description(f'{epoch} Test Loss: {stats.mean(ll)}')

            _, predicted = y.detach().max(1)

            for item in zip(predicted, target):
                confusion[item[0], item[1]] += 1

        precision_each, percent_correct = precision(confusion)
        precision_each = precision_each.numpy()
        percent_correct = np.array(percent_correct)
        precision_each = precision_each.reshape(1, -1)
        percent_correct = percent_correct.reshape(1, -1)
        precision_total = np.append(precision_each, percent_correct, axis=1)
        print(precision_total)
        columns = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'total']
        dt = pd.DataFrame(precision_total, columns=columns)
        dt.to_csv("precis_data2.csv", mode='a', index=False, header=None)

        classifier_save_path = Path('c:/data/deepinfomax/models/run' + str(run_id) + '/w_dim' + str(epoch) + '.mdl')
        classifier_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(classifier, str(classifier_save_path))
