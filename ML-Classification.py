# -*- coding: utf-8 -*-
import torch
import numpy as np

from torchvision import transforms

from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from PIL import Image
from os import path

np.random.seed(123)
torch.random.manual_seed(123)

# CLassification

def plot_logs(logs):
    training_losses, training_accuracies, test_losses, test_accuracies = \
    logs[0]['train'], logs[1]['train'], logs[0]['test'], logs[1]['test']
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(training_losses)
    plt.plot(test_losses)
    plt.legend(['Training Loss','Test Losses'])
    plt.grid()
    plt.subplot(122)
    plt.plot(training_accuracies)
    plt.plot(test_accuracies)
    plt.legend(['Training Accuracy','Test Accuracy'])
    plt.grid()
    plt.show()

class LocalDataset(Dataset):
    def __init__(self, base_path, txt_list, transform=None, reg=False):
        self.base_path=base_path
        self.data = np.loadtxt(txt_list,dtype=str,delimiter=',')
        self.transform = transform
        self.reg = reg

    def __getitem__(self, index):
        img_path, x, y, u, v, c = self.data[index]

        im = Image.open(path.join(self.base_path, img_path))

        if self.transform is not None:
            im = self.transform(im)

        if self.reg == False:
            label = int(c)
        else:
            label = torch.FloatTensor([float(x), float(y), float(u), float(v)])

        return {'image' : im, 'label': label}


    def __len__(self):
        return len(self.data)

class MiniAlexNetV3(nn.Module):
    def __init__(self, input_channels=3, out_classes=16):
        super(MiniAlexNetV3, self).__init__()

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
            nn.Dropout(),
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
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x

def train_classification(model, lr=0.01, epochs=20, momentum=0.9, weight_decay = 0.000001, train_loader=0, test_loader=0, type_m="classification"):
    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)

    loaders = {'train':train_loader, 'test':test_loader}
    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}
    preds = {'train':[], 'test':[]}
    if torch.cuda.is_available():
        model = model.cuda()

    for e in range(epochs):
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            outputs = []
            targets = []
            for i, batch in enumerate(loaders[mode]):
                # tensor -> variables
                x=Variable(batch['image'], requires_grad=True)
                y=Variable(batch['label'])

                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                output = model(x)

                l = criterion(output, y)
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                acc = accuracy_score(y.data,output.max(1)[1].data)

                epoch_loss += l.data[0] * x.shape[0]
                epoch_acc += acc * x.shape[0]
                samples += x.shape[0]

                print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" %  \
                        (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples),

            # if e == len(range(epochs)) -1:
            #     cm =  confusion_matrix(y.data,output.max(1)[1].data)
            #     score = f1_score(y.data,output.max(1)[1].data, average = None)
            #
            #     print i
            #     print len(loaders[mode])-1
            #     print "\n Confusion Matrix:"
            #     print cm
            #     print "\n F1 score: ", (score) , "\n"

            epoch_loss/=samples
            losses[mode].append(epoch_loss)
            epoch_acc/=samples
            accuracies[mode].append(epoch_acc)

            print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" %  \
                        (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc),
            print "\n"

            # save model
            torch.save(model.state_dict(),'c_model-%d.pth'%(e+1,))

            #print next(loaders['test'])
            # if e == len(range(epochs)) -1:
            # #     print Variable(loaders['test'])
            # #     print Variable(loaders['test']).shape
            #
            #     cm =  confusion_matrix(targets,outputs.max(1)[1])
            #     # score = f1_score(y.data,output.max(1)[1].data, average = None)
            #     # print i
            #     # print len(loaders[mode])-1
            #     print "\nconfusion matrix: \n"
            #     print cm
            #     # print "F1 score: ", (score)

    return model, (losses, accuracies)

train_set = LocalDataset('prj_dataset/images','prj_dataset/training_list.txt',transform=transforms.ToTensor(), reg=False)
test_set = LocalDataset('prj_dataset/images','prj_dataset/validation_list.txt',transform=transforms.ToTensor(), reg=False)

# mean, dev st. -> normalization
# mean
m = np.zeros(3)
for sample in train_set:
    m += sample['image'].sum(1).sum(1) # sum pixels per channels
m = m/(len(train_set)*256*256) # num_img * num_pixels

# dev st.
s = np.zeros(3)
for sample in train_set:
    s += ((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1)
s = np.sqrt(s/(len(train_set)*256*256))

transform_prj = transforms.Compose([transforms.RandomVerticalFlip(),
                                    transforms.ColorJitter(),
                                    transforms.RandomResizedCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize(m,s)
                                   ])

mini_alexnet_v3_class = MiniAlexNetV3(out_classes=16)

if path.isfile('c_model.pth'):
    mini_alexnet_v3_class.load_state_dict(torch.load('c_model.pth'))

train_set = LocalDataset('prj_dataset/images','prj_dataset/training_list.txt',transform=transform_prj, reg=False)
test_set = LocalDataset('prj_dataset/images','prj_dataset/validation_list.txt',transform=transform_prj, reg=False)

train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, num_workers=2)

mini_alexnet_v3_class, mini_alexnet_v3_class_logs = train_classification(mini_alexnet_v3_class, \
                                                                   train_loader=train_loader, \
                                                                   test_loader=test_loader, \
                                                                   epochs=2)
plot_logs(mini_alexnet_v3_class_logs)
