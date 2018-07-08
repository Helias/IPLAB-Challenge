# -*- coding: utf-8 -*-
import torch
import numpy as np

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR100
from torchvision import transforms

from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from PIL import Image
from os import path

np.random.seed(123)
torch.random.manual_seed(123)

class MiniAlexNetV3(nn.Module):
    def __init__(self, input_channels=3, out_classes=16):
        super(MiniAlexNetV3, self).__init__()
        #ridefiniamo il modello utilizzando i moduli sequential.
        #ne definiamo due: un "feature extractor", che estrae le feature maps
        #e un "classificatore" che implementa i livelly FC
        self.feature_extractor = nn.Sequential(
            #Conv1
            nn.Conv2d(input_channels, 16, 5, padding=2), #Input: 3 x 28 x 28. Ouput: 16 x 28 x 28
            nn.MaxPool2d(2), #Input: 16 x 28 x 28. Output: 16 x 14 x 14
            nn.ReLU(),

            #Conv2
            nn.Conv2d(16, 32, 5, padding=2), #Input 16 x 14 x 14. Output: 32 x 14 x 14
            nn.MaxPool2d(2), #Input: 32 x 14 x 14. Output: 32 x 7 x 7
            nn.ReLU(),

            #Conv3
            nn.Conv2d(32, 64, 3, padding=1), #Input 32 x 7 x 7. Output: 64 x 7 x 7
            nn.ReLU(),

            #Conv4
            nn.Conv2d(64, 128, 3, padding=1), #Input 64 x 7 x 7. Output: 128 x 7 x 7
            nn.ReLU(),

            #Conv5
            nn.Conv2d(128, 256, 3, padding=1), #Input 128 x 7 x 7. Output: 256 x 7 x 7
            nn.MaxPool2d(2), #Input: 256 x 7 x 7. Output: 256 x 3 x 3
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(), #i layer di dropout vanno posizionati prima di FC6 e FC7
            #FC6
            nn.Linear(2304, 2048), #Input: 256 * 3 * 3
            nn.ReLU(),

            nn.Dropout(),
            #FC7
            nn.Linear(2048, 1024),
            nn.ReLU(),

            #FC8
            nn.Linear(1024, out_classes)
        )


    def forward(self,x):
        #Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x

class ScenesDataset2(Dataset):
    """Implementa l'oggetto ScenesDataset che ci permette di caricare
    le immagini del dataset 8 Scenes"""
    def __init__(self,base_path,txt_list,transform=None):
        """Input:
            base_path: il path alla cartella contenente le immagini
            txt_list: il path al file di testo contenente la lista delle immagini
                        con le relative etichette. Ad esempio train.txt o test.txt.
            transform: implementeremo il dataset in modo che esso supporti le trasformazioni"""
        #conserviamo il path alla cartella contenente le immagini
        self.base_path=base_path
        #carichiamo la lista dei file
        #sarà una matrice con n righe (numero di immagini) e 2 colonne (path, etichetta)
        self.data = np.loadtxt(txt_list,dtype=str,delimiter=',')
        #conserviamo il riferimento alla trasformazione da applicare
        self.transform = transform

    def __getitem__(self, index):
        #recuperiamo il path dell'immagine di indice index e la relativa etichetta
        f, x, y, u, v, c = self.data[index]

        #carichiamo l'immagine utilizzando PIL
        im = Image.open(path.join(self.base_path, f))

        #se la trasfromazione è definita, applichiamola all'immagine
        if self.transform is not None:
            im = self.transform(im)

        #convertiamo l'etichetta in un intero
        label = int(c)

        #restituiamo un dizionario contenente immagine etichetta
        return {'image' : im, 'label': label}

    #restituisce il numero di campioni: la lunghezza della lista "images"
    def __len__(self):
        return len(self.data)

def train_classification(model, lr=0.01, epochs=20, momentum=0.9, weight_decay = 0.000001, train_loader=0, test_loader=0):


    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)

    loaders = {'train':train_loader, 'test':test_loader}
    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}
    cms = {'train':[], 'test':[]}
    if torch.cuda.is_available():
        model=model.cuda()

    for e in range(epochs):
        for mode in ['train', 'test']:
            if mode=='train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            for i, batch in enumerate(loaders[mode]):
                #trasformiamo i tensori in variabili
                x=Variable(batch['image'], requires_grad=True)
                y=Variable(batch['label'])

                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                output = model(x)
                l = criterion(output,y)

                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                acc = accuracy_score(y.data,output.max(1)[1].data)
                cm =  confusion_matrix(y.data,output.max(1)[1].data)

                epoch_loss+=l.data[0]*x.shape[0]
                epoch_acc+=acc*x.shape[0]
                samples+=x.shape[0]

                print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" %  \
                        (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples),



            epoch_loss/=samples
            epoch_acc/=samples

            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)
            cms[mode].append(cm)
            print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" %  \
                        (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc)

    return model, (losses, accuracies)


train_set2 = ScenesDataset2('prj_dataset/images','prj_dataset/training_list.txt',transform=transforms.ToTensor())
test_set2 = ScenesDataset2('prj_dataset','prj_dataset/validation_list.txt',transform=transforms.ToTensor())

m2 = np.zeros(3)
for sample in train_set2:
    m2 += sample['image'].sum(1).sum(1) #accumuliamo la somma dei pixel canale per canale

#dividiamo per il numero di immagini moltiplicato per il numero di pixel
m2 = m2/(len(train_set2)*256*256)

#procedura simile per calcolare la deviazione standard
s2 = np.zeros(3)
for sample in train_set2:
    s2 += ((sample['image']-torch.Tensor(m2).view(3,1,1))**2).sum(1).sum(1)

s2 = np.sqrt(s2/(len(train_set2)*256*256))


transform_prj = transforms.Compose([transforms.RandomVerticalFlip(),
                                    transforms.ColorJitter(),
                                    transforms.RandomResizedCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize(m2,s2)
                                   ])

train_set = ScenesDataset2('prj_dataset/images','prj_dataset/training_list.txt',transform=transform_prj)
test_set = ScenesDataset2('prj_dataset/images','prj_dataset/validation_list.txt',transform=transform_prj)

train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, num_workers=2)

mini_alexnet_v3_cifar = MiniAlexNetV3()
if path.isfile('model.pth'):
    mini_alexnet_v3_cifar.load_state_dict(torch.load('model.pth'))

mini_alexnet_v3_cifar, mini_alexnet_v3_cifar_logs = train_classification(mini_alexnet_v3_cifar, \
                                                                   train_loader=train_loader, \
                                                                 test_loader=test_loader, epochs=30)
