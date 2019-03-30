from dataset import torch, os, LocalDataset, transforms, np, Image
from config import *

from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import resnet, vgg

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
import gc

# directory results
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

cuda_available = torch.cuda.is_available()

# Load dataset

# pre-computed mean and standard_deviation
mean = torch.Tensor([0.3877, 0.3647, 0.3547])
std_dev = torch.Tensor([0.2121, 0.2106, 0.2119])

transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std_dev)])

training_set = LocalDataset(IMAGES_PATH, TRAINING_PATH, transform=transform)
validation_set = LocalDataset(IMAGES_PATH, VALIDATION_PATH, transform=transform)
test_set = LocalDataset(IMAGES_PATH, TEST_PATH, transform=transform)

training_set_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=True)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False)
test_set_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False)

if REGRESSION:
    classes = {"num_classes": 4}
else:
    classes = {"num_classes": 16}

def train_model(model_name, model, lr=LEARNING_RATE, epochs=EPOCHS, momentum=MOMENTUM, weight_decay=0, train_loader=training_set_loader, test_loader=validation_set_loader):
    
    if not os.path.exists(RESULTS_PATH + "/" + model_name):
        os.makedirs(RESULTS_PATH + "/" + model_name)
    
    if REGRESSION:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    loaders = {'train':train_loader, 'test':test_loader}
    losses = {'train':[], 'test':[]}
    accuracies = {'train':[], 'test':[]}

    #testing variables
    y_testing = []
    preds = []

    if USE_CUDA and cuda_available:
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

                # convert tensor to variable
                x=Variable(batch['image'], requires_grad=(mode=='train'))
                y=Variable(batch['label'])

                if USE_CUDA and cuda_available:
                    x = x.cuda()
                    y = y.cuda()

                output = model(x)
                l = criterion(output, y) # loss

                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    y_testing.extend(y.data.tolist())
                    preds.extend(output.max(1)[1].tolist())

                if USE_CUDA and cuda_available:
                    acc = accuracy_score(y.data.cuda().cpu().numpy(), output.max(1)[1].cuda().cpu().numpy())
                else:
                    acc = accuracy_score(y.data, output.max(1)[1])

                epoch_loss += l.data.item()*x.shape[0] # l.data[0]
                epoch_acc += acc*x.shape[0]
                samples += x.shape[0]

                print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
                    (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples))

                #debug
                if DEBUG and i == 2:
                  break

            epoch_loss /= samples
            epoch_acc /= samples
            
            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)


            print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
                  (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc))
            
    torch.save(model.state_dict(), str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + ".pt")
    return model, (losses, accuracies), y_testing, preds


def test_model(model_name, model, test_loader=validation_set_loader):
    model.load_state_dict(torch.load(str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + ".pt"))

    if USE_CUDA and cuda_available:
        model = model.cuda()

    model.eval()

    preds = []
    gts = []
    
    #debug
    i = 0
    
    for batch in test_loader:
        x = Variable(batch['image'])
        if USE_CUDA and cuda_available:
            x = x.cuda()
            pred = model(x).data.cuda().cpu().numpy().copy()
        else:
            pred = model(x).data.numpy().copy()

        gt = batch['label'].numpy().copy()
        preds.append(pred)
        gts.append(gt)
        
        # debug
        if DEBUG and i == 2:
            break
        else:
          i+=1
        
    return np.concatenate(preds), np.concatenate(gts)

def write_stats(model_name, Y, predictions, gts, predictions2):
    if not os.path.exists(RESULTS_PATH + "/" + model_name):
        os.makedirs(RESULTS_PATH + "/" + model_name)

    acc = accuracy_score(gts, predictions2.argmax(1))
    cm = confusion_matrix(Y, predictions)

    if DEBUG:
      score = "00 F1_SCORE 00"
    else:
      score = f1_score(Y, predictions, average=None)
    
    file = open(str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + "_stats.txt", "w+")
    file.write ("Accuracy: " + str(acc) + "\n\n")
    file.write("Confusion Matrix: \n" + str(cm) + "\n\n")
    file.write("F1 Score: \n" + str(score))
    file.close()

def plot_logs_classification(model_name, logs):
    if not os.path.exists(RESULTS_PATH + "/" + model_name):
        os.makedirs(RESULTS_PATH + "/" + model_name)

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
    #plt.show()
    plt.savefig(str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + "_graph.png")

def train_model_iter(model_name, model, weight_decay=0):
    model, loss_acc, y_testing, preds = train_model(model_name=model_name, model=model, weight_decay=weight_decay)
    preds_test, gts = test_model(model_name, model=model)
    write_stats(model_name, y_testing, preds, gts, preds_test)
    plot_logs_classification(model_name, loss_acc)
    gc.collect()

def generate_test(model_name, model):
    model.load_state_dict(torch.load(str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + ".pt"))

    # test_list = open("test.csv", "r").read().split("\n")
    test_list = open(TEST_PATH, "r").read().split("\n")
    test_list.remove('')

    output_test = ""

    for img in test_list:
        image_path = "images/"+str(img)
        im = Image.open(image_path).convert("RGB")
        im = transform(im)

        if USE_CUDA and cuda_available:
            model = model.cuda()
        model.eval()

        x = Variable(im.unsqueeze(0))

        if USE_CUDA and cuda_available:
            x = x.cuda()
            pred = model(x).data.cuda().cpu().numpy().copy()
        else:
            pred = model(x).data.numpy().copy()

        idx_max_pred = np.argmax(pred)
        idx_classes = idx_max_pred % 16

        output_test += str(img)+","+str(idx_classes)+"\n"
    
    f = open("output.csv", "w+")
    f.write(output_test)
    f.close()


# Resnet 152
resnet152_model = resnet.resnet152(pretrained=True)
resnet152_model.fc = nn.Linear(512, classes["num_classes"]) # change num_classes to pretrained model
# train_model_iter("resnet152", resnet152_model)

# VGG 19
vgg19_model = vgg.vgg19(pretrained=True)
vgg19_model.classifier[6] = nn.Linear(4096, classes["num_classes"]) # change num_classes to pretrained model
# train_model_iter("resnet152", resnet152_model)


# generate_test("resnet152", resnet152_model)

# Regularization

# # Weight Decay
# resnet152_model = resnet.resnet152(pretrained=True, **classes)
# train_model_iter("resnet152_wd", resnet152_model, weight_decay=WEIGHT_DECAY)

# # Data Augmentation
# transform = transforms.Compose([transforms.RandomVerticalFlip(),
#                                 transforms.ColorJitter(),
#                                 transforms.RandomCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean, std_dev)])

# training_set = LocalDataset(IMAGES_PATH, TRAINING_PATH, transform=transform)
# validation_set = LocalDataset(IMAGES_PATH, VALIDATION_PATH, transform=transform)
# test_set = LocalDataset(IMAGES_PATH, TEST_PATH, transform=transform)

# training_set_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=True)
# validation_set_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False)
# test_set_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=THREADS, shuffle=False)

# train_model_iter("resnet152_da", resnet152_model)
# train_model_iter("resnet152_da_wd", resnet152_model, weight_decay=WEIGHT_DECAY)

