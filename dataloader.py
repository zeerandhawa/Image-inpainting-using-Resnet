import os
from os.path import isdir, exists, abspath, join
import torch
import random
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import PIL
# import cv2
import random


class DataLoader():
    def __init__(self, root_dir='data', batch_size=16, test_percent=.20):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)


        self.files = os.listdir(self.root_dir)
        # because file names are same for both images and labels

        self.data_files = [join(self.root_dir, f) for f in self.files]
        # self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            #endId = n_train
            endId = 100
        elif self.mode == 'test':
            current = n_train
            current = 0
            endId = len(self.data_files)
            self.batch_size = 1
            #print('test batch size' + str(self.batch_size))

        while current < endId:
            current += 1

            image_batch = []
            label_batch = []

            # todo: load images and labels
            if (self.mode == 'test'):
                print('inside while')
            for bt in range(0, self.batch_size):
                if (self.mode == 'test'):
                    print('inside for')
                augment_choice = np.random.randint(0, 15)
                #augment_choice = 1

                image_path = self.data_files[0]
                # label_path = self.label_files[current-1]
                im = Image.open(image_path)

                # random cropping a 128*128 patch

                rancr = transforms.RandomCrop([128, 128])
                image = rancr(im)

                #toTensor normalizes the image between 0 and 1

                label = image

                # if (augment_choice == 1):
                #     image = image.convert('L')
                #     enhancer = PIL.ImageEnhance.Brightness(image)
                #     image = enhancer.enhance(2.0)
                #     label = label.convert('L')
                #     enhancer = PIL.ImageEnhance.Brightness(label)
                #     label = enhancer.enhance(2.0)

                if (augment_choice == 2):
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    label = label.transpose(Image.FLIP_TOP_BOTTOM)
                if (augment_choice == 3):
                    image = image.rotate(90)
                    label = label.rotate(90)
                if (augment_choice == 4):
                    image = image.transpose(Image.ROTATE_270)
                    label = label.transpose(Image.ROTATE_270)
                if (augment_choice == 5 | augment_choice == 6 | augment_choice == 7):
                    clrjitt = transforms.ColorJitter(hue=0.5)

                    image = clrjitt(image)
                    label = image
                    toTensor = transforms.ToTensor()
                    #image = toTensor(image)
                    #label = toTensor(label)

                toTensor = transforms.ToTensor()
                image = toTensor(image)
                label = toTensor(label)

                image = image.cuda()
                #print('size of image in dl:' + str(image.shape))

                im = self.provideMask(image)
                im = im.cpu()

                #print('size of image in dl after mask:' + str(im.shape))
                im = im.view(1, 4, 128, 128)
                label = label.view(1, 3, 128, 128)

                # # hint: scale images between 0 and 1
                # data_image = data_image / 255
                # label_image = label_image / 255

                if (bt == 0):
                    #print ('bt' + str(bt))
                    image_batch = im
                    label_batch = label

                    # print('image batch type'+str(type(image_batch)))
                else:
                    #print ('bt' + str(bt))
                    image_batch = torch.cat((image_batch, im), 0)
                    label_batch = torch.cat((label_batch, label), 0)

            # transform = transforms.Compose([])

            # hint: if training takes too long or memory overflow, reduce image size!
            # print('image batch size' + str(image_batch.size()))
            # print('label batch size' + str(label_batch.size()))

            image_batch = np.asarray(image_batch)
            label_batch = np.asarray(label_batch)
            # print('image batch type after loop' + str(type(image_batch)))

            # hint: scale images between 0 and 1
            #print ('image_batch in dataloader')
            #print (image_batch)
            #image_batch = image_batch / 255
            #label_batch = label_batch / 255
            yield (image_batch, label_batch)

    def provideMask(self, image):
        mask = torch.ones([128, 128])

        for i in range(0, 5):
            x = np.random.randint(0, 118)
            y = np.random.randint(0, 63)
            box = torch.zeros([8, 64])
            mask[x:x + 8, y:y + 64] = box
            image[:, x:x + 8, y:y + 64] = box

        mask = mask.reshape(1, 128, 128)

        #print('image size' + str(image.size()))
        #print('mask size' + str(mask.size()))
        image = torch.cat((image, mask), 0)
        #image = torch.cat((mask, image), 0)
        #print('size after concat' + str(image.size()))
        return image.cuda()

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))