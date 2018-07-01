import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader import ImageDataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt
import torch.nn as nn
from tensorboardX import SummaryWriter

img_height = 224
img_width = 224
batch_size = 16
learning_rate = 0.00025
epochs = 50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))

    return writer


transforms_ = [transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
               transforms.RandomCrop((img_height, img_width)),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

train_loader = DataLoader(ImageDataset("E:\\Datasets\\NUMTA", transforms_=transforms_, mode='train'),
                          batch_size=batch_size, shuffle=True)

test_loader = DataLoader(ImageDataset("E:\\Datasets\\NUMTA", transforms_=transforms_, mode='test'),
                         batch_size=batch_size, shuffle=True)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

writer = create_summary_writer(model, train_loader, './logdir/pretrained_resnet50_lr_0.00025_small_batch')
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    total = 0
    correct = 0

    for i, (x, y) in enumerate(train_loader):
        images = Variable(x.type(torch.FloatTensor))
        labels = Variable(y.type(torch.LongTensor))

        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}'
                  .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), (100 * correct / total)))
            writer.add_scalar("training/loss", loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("training/accuracy", 100 * correct / total, epoch * len(train_loader) + i)

    # test the model
    total = 0
    correct = 0

    for i, (x, y) in enumerate(test_loader):
        images = Variable(x.type(torch.FloatTensor))
        labels = Variable(y.type(torch.LongTensor))

        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    writer.add_scalar("test/accuracy", 100 * correct / total, epoch)
    print("EPOCH: ", epoch, "   Accuracy: ", 100 * correct / total)
    torch.save(model.state_dict(), './saved_models/model.ckpt')
