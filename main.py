import livelossplot
import torch
import cv2
import torchvision
from PIL import ImageFilter
from numpy import random
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys, tarfile, errno
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchsummary import summary
import time
import threading
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

#################### ML Device #######################################
device = 'cuda'
print('Chosen device : ')
print(device)

#################### Dataset Functions ###############################

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH_TRAIN = './data/stl10_binary/train_X.bin'
DATA_PATH_TEST = './data/stl10_binary/test_X.bin'

# path to the binary train file with labels
LABEL_PATH_TRAIN = './data/stl10_binary/train_y.bin'
LABEL_PATH_TEST = './data/stl10_binary/test_y.bin'


def read_labels(path_to_labels):
    """
  :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
  :return: an array containing the labels
  """
    print('Reading labels from : %s', path_to_labels)
    with open(path_to_labels, 'rb') as f:
        dataset_y = np.fromfile(f, dtype=np.uint8)
        return dataset_y


def read_images(path_to_data):
    """
  :param path_to_data: the file containing the binary images from the STL-10 dataset
  :return: an array containing all the images
  """

    print('Reading files from : %s', path_to_data)
    with open(path_to_data, 'rb') as f:
        # read file
        input = np.fromfile(f, dtype=np.uint8)

        # Force numpy with -1 to determine size
        # Image = 96x96x3, RGB, column major order
        dataset_x = np.reshape(input, (-1, 3, 96, 96))

        # Merge channels = can grayscale it later
        # images = np.transpose(images, (0, 3, 2, 1))
        return dataset_x


def read_single_image(image_file):
    """
  CAREFUL! - this method uses a file as input instead of the path - so the
  position of the reader will be remembered outside of context of this method.
  :param image_file: the open file containing the images
  :return: a single image
  """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    print(image)
    image = image.astype(np.float32)
    print(image)
    image = image / 255.
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
  :param image: the image to be plotted in a 3-D matrix format
  :return: None
  """
    plt.imshow(image)
    plt.show()


def normalize_min_max(x, new_min=0.0, new_max=1.0):
    min_val = np.min(x, 0, keepdims=True)
    max_val = np.max(x, 0, keepdims=True)
    x = (x - min_val) / (max_val - min_val + 1e-12)
    x = x * (new_max - new_min) + new_min
    return x


def show_some_images(x, sqrtN, fig_num=99):
    plt.figure(fig_num)
    plt.clf()
    i = 1
    for n1 in range(1, sqrtN + 1):
        for n2 in range(1, sqrtN + 1):
            plt.subplot(sqrtN, sqrtN, i)
            plt.imshow(np.reshape(x[i - 1, :], (96, 96)), cmap='gray')
            plt.axis('off')
            i += 1


def apply_gauss(in_image, ksizex, ksizey):
    x = cv2.GaussianBlur(in_image, (ksizex, ksizey), 0)
    return x


def compare_gauss():
    with open(DATA_PATH_TRAIN) as f:
        image = read_single_image(f)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(image)
        ax.set_title('Before')
        image = apply_gauss(image, 3, 3)

        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(image)
        ax.set_title('After')
        plt.show()


def create_blur_set(in_dataset, ksizex, ksizey):
    output_dataset = np.zeros(in_dataset.shape)
    tqdm_list = list(range(in_dataset.shape[0]))
    print('Starting Blur')
    for i in tqdm.tqdm(tqdm_list):
        output_dataset[i, :, :, :] = cv2.GaussianBlur(in_dataset[i, :, :, :], (ksizex, ksizey), 0)
    # print(i)
    print('Done Blurring')
    return output_dataset

def check_pairs(dataLoader):
    for i, data in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size, position=0,
                             leave=True):
        blur_img = data[0]
        #print(blur_img[0,:,:,:])
        sharp_img = data[1]
        image = blur_img[1]
        image = np.transpose(image, (2, 1, 0))
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(image)
        ax.set_title('Before - Blurred')
        image = sharp_img[1]
        image = np.transpose(image, (2, 1, 0))

        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(image)
        ax.set_title('After - Deblurred')
        plt.show()
        time.sleep(4)


#################### Model Helper Functions ##########################
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('New checkpoint in:  %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('Loaded checkpoint from: %s' % checkpoint_path)


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )


def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
    conv = nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    torch.cat(conv, in_fine)

    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    )
    upsample(in_coarse)


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


######################## MODEL ###################################################

class Net(nn.Module):
    def __init__(self, useBN=False):
        super(Net, self).__init__()

        self.conv1 = add_conv_stage(1, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)
        self.conv5 = add_conv_stage(256, 512, useBN=useBN)

        self.conv4m = add_conv_stage(512, 256, useBN=useBN)
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128, 64, useBN=useBN)
        self.conv1m = add_conv_stage(64, 32, useBN=useBN)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            #nn.Sigmoid()
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        conv1_out = self.conv1(x)
        # return self.upsample21(conv1_out)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)

        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        conv0_out = self.conv0(conv1m_out)

        return conv0_out


def model_RGB(model, img_in):
  img_R = model(img_in[:, 0, :, :].unsqueeze(1))
  img_G = model(img_in[:, 1, :, :].unsqueeze(1))
  img_B = model(img_in[:, 2, :, :].unsqueeze(1))
  img_out = torch.cat((img_R, img_G, img_B), dim=1)
  #print(img_out.shape)
  return img_out

class GaussDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None):
        self.X = blur_paths
        self.y = sharp_paths

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        blur_image = self.X[i, :, :, :]
        blur_image = torch.from_numpy(blur_image)

        if self.y is not None:
            sharp_image = self.y[i, :, :, :]
            sharp_image = torch.from_numpy(sharp_image)
            return blur_image, sharp_image
        else:
            return blur_image

def create_dataset(batch_size = 8, noise_level=0.0):
    images_train = read_images(DATA_PATH_TRAIN)
    print(images_train.shape)
    print(type(images_train))

    labels_train = read_labels(LABEL_PATH_TRAIN)
    print(labels_train.shape)

    images_test = read_images(DATA_PATH_TEST)
    print(images_train.shape)

    labels_test = read_labels(LABEL_PATH_TEST)
    print(labels_test.shape)

    #images_train_blur = create_blur_set(images_train, kx, ky)


    images_train = normalize_min_max(images_train, 0.0, 1.0)

    images_train_blur = np.add(images_train, noise_level * np.random.normal(loc=0, scale=(np.max(images_train) - np.min(images_train)) / 6., size=images_train.shape))
    images_train_blur = normalize_min_max(images_train_blur, 0.0, 1.0)
    print(images_train_blur.shape)

    images_train = images_train.astype(np.float32)
    images_train_blur = images_train_blur.astype(np.float32)

    (x_train, x_val, y_train, y_val) = train_test_split(images_train_blur, images_train, test_size=0.25)
    #print("Data shapes:")
    #print(f"x_train: {x_train.shape}")
    #print(f"x_val: {x_val.shape}")
    #print(f"y_train: {y_train.shape}")
    #print(f"y_val: {y_val.shape}")
    #print('Done shapes')

    train_data = GaussDataset(x_train, y_train)
    train_Loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = GaussDataset(x_val, y_val)
    val_Loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_Loader, val_Loader

def initialize_model(train=False):
    model = Net().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )
    summary(model, (1, 96, 96))

    if train:
        model.train()
    else:
        model.eval()
        load_checkpoint('./checkpoints/latestModel.pth', model, optimizer)

    return model, criterion, lr_scheduler, optimizer


def train_step(model, dataLoader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    print(' ')
    print('Train Step:', epoch)
    for i, data in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size, position=0,leave=True):
    #for i, data in enumerate(dataLoader):
        blur_img = data[0].to(device)
        sharp_img = data[1].to(device)
        optimizer.zero_grad()
        output = model_RGB(model, blur_img)
        loss = criterion(output, sharp_img)
        #print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(dataLoader.dataset)
    #print(f"Train Loss: {train_loss:.5f}")

    return train_loss


def val_step(model, dataLoader, optimizer, criterion, epoch):
    model.eval()
    running_loss = 0.0
    #print(' ')
    #print('Eval Step')
    with torch.no_grad():
        #for i, data in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size, position=0, leave=True):
        for i, data in enumerate(dataLoader):
            blur_img = data[0].to(device)
            sharp_img = data[1].to(device)
            output = model_RGB(model, blur_img)
            samples = torch.stack([blur_img[0], output[0], sharp_img[0]], dim=0)
            loss = criterion(output, sharp_img)
            running_loss += loss.item()
        val_loss = running_loss / len(dataLoader.dataset)
        #print(f"Val Loss: {val_loss:.5f}")

        return val_loss, samples

def train(nr_epoch, model, trainLoader, valLoader, optimizer, criterion, lr_scheduler):
    train_loss = []
    val_loss = []
    min_loss = sys.float_info.max
    tb = SummaryWriter()
    start = time.time()
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    last_ck = './checkpoints/Model_' + current_time + '.pth'
    print("Check . tensorboard --logdir=runs . for logs")

    for epoch in range(nr_epoch):
        logs = {}
        train_epoch_loss = train_step(model, trainLoader, optimizer, criterion, epoch)
        val_epoch_loss, samples = val_step(model, valLoader, optimizer, criterion, epoch)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        lr_scheduler.step(val_epoch_loss)


        #plot_loss(train_loss, val_loss)
        tb.add_scalar("Train Loss", train_epoch_loss, epoch)
        tb.add_scalar("Val Loss", val_epoch_loss, epoch)
        #tb.add_image('Input - Gauss', samples[0])
        #tb.add_image('Output', samples[1])
        #tb.add_image('Original -
        print(samples.shape)
        img_grid = torchvision.utils.make_grid(samples)
        tb.add_image("Input - Output - Original", img_grid)

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
        print(f"################################ \n")
        print(f"Epoch Losses - Min Val: {min_loss:.5f} \n")
        print(f"Train: {train_epoch_loss:.5f} & Val: {val_epoch_loss:.5f} \n")
        save_checkpoint(last_ck, model, optimizer)
        print('Saved Checkpoint!')
        print(f"################################ \n")

    end = time.time()
    tb.close()
    print(f"Took {((end - start) / 60):.3f} minutes to train")

def plot_loss(train_loss, val_loss):
     plt.figure(figsize=(10, 7))
     plt.plot(train_loss, color='orange', label='train loss')
     plt.plot(val_loss, color='red', label='validataion loss')
     plt.xlabel('Epochs')
     plt.ylabel('Loss')
     plt.savefig('loss.png')


def visual_test(model, optimizer, checkpoint, dataLoader):
    load_checkpoint(checkpoint, model, optimizer)
    model.eval()
    print('Visual Test!')
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size,
                                 position=0, leave=True):
            blur_img = data[0].to(device)
            sharp_img = data[1].to(device)
            output = model_RGB(model, blur_img)

            for index in range(dataLoader.batch_size):
                sample = blur_img[index].cpu()
                sample = np.transpose(sample, (2, 1, 0))
                fig = plt.figure()
                ax = fig.add_subplot(1, 3, 1)
                imgplot = plt.imshow(sample)
                ax.set_title('Before - Blurred')

                sample = output[index].cpu()
                sample = np.transpose(sample, (2, 1, 0))
                ax = fig.add_subplot(1, 3, 2)
                imgplot = plt.imshow(sample)
                ax.set_title('Predicted - Sharp')

                sample = sharp_img[index].cpu()
                sample = np.transpose(sample, (2, 1, 0))
                ax = fig.add_subplot(1, 3, 3)
                imgplot = plt.imshow(sample)
                ax.set_title('Target - Sharp')
                plt.show()
                time.sleep(4)

def plot_sample(sample1, sample2, sample3, epoch):
    sample1 = np.transpose(sample1, (2, 1, 0))
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(sample1)
    ax.set_title('Before - Blurred')

    sample2 = np.transpose(sample2, (2, 1, 0))
    ax = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(sample2)
    ax.set_title('Predicted - Sharp')

    sample3 = np.transpose(sample3, (2, 1, 0))
    ax = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(sample3)
    ax.set_title('Target - Sharp')
    plt.savefig("./comparisons/Comparison-Epoch" + str(epoch) + ".png")
    plt.close("all")

if __name__ == '__main__':
    nr_epoch = 100
    #matplotlib.use('Agg')
    trainLoader, valLoader = create_dataset(batch_size=8, noise_level=0.25)
    #check_pairs(trainLoader)

    model, criterion, lr_scheduler, optimizer = initialize_model(train=True)
    train(nr_epoch, model, trainLoader, valLoader, optimizer, criterion, lr_scheduler)
    #compare_gauss()
    #checkpoint = './checkpoints/16-02-2022-DeblurSTD.pth'
    #visual_test(model, optimizer, checkpoint, valLoader)

    #image += noise_level * torch.normal(mean=0, std=(image.max() - image.min()) / 6., size=image.shape).cuda()
