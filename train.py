import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import ResNet
from dataloader import DataLoader


def train_net(net,
              epochs=5,
              data_dir='data/',
              n_classes=2,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              gpu=False):
    train_loader = DataLoader(join(data_dir, 'train'))
    test_loader = DataLoader(join(data_dir, 'test'))

    if(torch.cuda.is_available()):
        print('available')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')


    net = net.cuda()
    N_train = train_loader.n_train()

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum = 0.99,
                          weight_decay=0.0000005)
    print(net)
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        train_loader.setMode('train')

        epoch_loss = 0

        for i, (img, label) in enumerate(train_loader):
            print('epoch: ' + str(epoch))
            print('iteration: '+str(i))

            #label = label - 1
            optimizer.zero_grad()
            # todo: create image tensor: (N,C,H,W) - (batch size=16,channels=4,height,width)

            shape = img.shape
            img_batch = torch.from_numpy(img.reshape(16, 4, shape[2], shape[3])).float()



            label_batch = torch.from_numpy(label.reshape(16, 3, shape[2], shape[3])).float()
            label_batch = label_batch.cuda()
            img_batch = Variable(img_batch)
            label_batch = Variable(label_batch)

            # todo: load image tensor to gpu
            if gpu:
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            # todo: get prediction and getLoss()
            predicted = net.forward(img_batch)


            #loss = getLoss(predicted, label)
            loss = getLoss(label_batch, predicted)

            epoch_loss += loss.item()

            print('Training sample %d / %d - Loss: %.6f' % (i + 1, N_train, loss.item()))

            # optimize weights
            loss.backward()
            optimizer.step()

            if ((epoch % 5 == 0) and (i % 99 == 0)):
                test_loader.setMode('test')
                net.eval()
                print('test')
                with torch.no_grad():
                    for _, (img_val, label_val) in enumerate(test_loader):
                        print('inside test')
                        shape = img.shape

                        img_test = torch.from_numpy(img_val.reshape(1, 4, shape[2], shape[3])).float()
                        if gpu:
                            img_test = img_test.cuda()
                            img_test = img_test.cuda()
                        pred = net(img_test)

                        pred = pred.cpu().detach().numpy()
                        sub = pred - label_val
                        loss = np.mean(np.power(sub, 2))
                        print('test loss' + str(loss))
                        plt.subplot(1, 3, 1)
                        img_test = img_test.cpu().detach().numpy()
                        img = np.reshape(img_test, (4, 128, 128))
                        img = img[:3, :, :]
                        img = np.transpose(img, (1, 2, 0))

                        plt.imshow(img)
                        plt.title(str(epoch+1) + 'test-in')

                        plt.subplot(1, 3, 2)
                        label_val = np.reshape(label_val, (3, 128, 128))
                        label_val = np.transpose(label_val, (1, 2, 0))

                        plt.imshow(label_val)
                        plt.title(str(epoch+1) + 'test-gt')
                        plt.subplot(1, 3, 3)
                        pred = np.reshape(pred, (3, 128, 128))

                        pred = np.transpose(pred, (1, 2, 0))
                        plt.imshow(pred)
                        plt.title(str(epoch+1) + 'test-out')
                        plt.savefig(str(epoch+1) + '-test.png')
                        plt.show()




                        plt.subplot(1, 3, 1)
                        img_torch = img_batch.cpu().detach().numpy()
                        img_torch = img_torch[0,:, :, :]
                        img = np.reshape(img_torch, (4, 128, 128))
                        img = img[:3, :, :]
                        img = np.transpose(img, (1, 2, 0))

                        plt.imshow(img)
                        plt.title(str(epoch+1) + 'train-in')

                        plt.subplot(1, 3, 2)
                        label_val = label_batch.cpu().detach().numpy()
                        label_val = label_val[0,:, :, :]
                        label_val = np.reshape(label_val, (3, 128, 128))
                        label_val = np.transpose(label_val, (1, 2, 0))

                        plt.imshow(label_val)
                        plt.title(str(epoch+1) + 'train-gt')
                        plt.subplot(1, 3, 3)
                        predicted = predicted.cpu().detach().numpy()
                        pred = predicted[0,:, :, :]
                        pred = np.reshape(pred, (3, 128, 128))

                        pred = np.transpose(pred, (1, 2, 0))

                        plt.imshow(pred)
                        plt.title(str(epoch+1)+   'train-out')
                        plt.savefig(str(epoch+1)+'-train.png')
                        plt.show()



        if(epoch%99 == 0):
            torch.save(net.state_dict(), join(data_dir, 'Checkpoints') + '/CP%d.pth' % (epoch + 1))
            print('Checkpoint %d saved !' % (epoch + 1))

            print('CHECK DIVISION BY i, should be i intead of i+1')

            print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / (i+1)))


    # displays test images with original and predicted masks after training



def getLoss(label_batch, predicted):
    loss = ((label_batch - predicted) ** 2).mean()
    return loss




def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = ResNet()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    #PATH TO SAVED MODEL
    #net.load_state_dict(torch.load('/home/skrandha/sfuhome/UNetFW/data/cells/good_model/CP5.pth'))
    train_net(net=net,
              epochs=args.epochs,
              n_classes=args.n_classes,
              gpu=args.gpu,
              data_dir=args.data_dir)

