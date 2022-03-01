import livelossplot
import tensorboard.summary
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
from torchinfo import summary
import time
import threading
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import base64
import hashlib
#from Crypto import Random
#from Crypto.Cipher import AES

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
    min_val = np.min(x)
    #print("Min", min_val)
    max_val = np.max(x)
    #print("Max", max_val)
    x = (x - min_val) / (max_val - min_val)
    x = x * (new_max - new_min) + new_min
    #x = (x - np.min(x))/np.ptp(x)
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

# class AESCipher(object):
#
#     def __init__(self, key):
#         self.bs = 32
#         self.key = hashlib.sha256(key.encode()).digest()
#
#     def encrypt(self, raw):
#         raw = self._pad(raw)
#         iv = Random.new().read(AES.block_size)
#         cipher = AES.new(self.key, AES.MODE_CBC, iv)
#         return base64.b64encode(iv + cipher.encrypt(raw))
#
#     def decrypt(self, enc):
#         enc = base64.b64decode(enc)
#         iv = enc[:AES.block_size]
#         cipher = AES.new(self.key, AES.MODE_CBC, iv)
#         return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')
#         #return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')
#
#     def _pad(self, s):
#         #print("25")
#         #print(type(s))
#         #print(type((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)))
#         return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs).encode('utf-8')
#
#     @staticmethod
#     def _unpad(s):
#         return s[:-ord(s[len(s)-1:])]


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
  img_out = torch.clamp(img_out, 0., 1.)
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


def create_dataset_noise(tb, batch_size=8, noise_level_min=0.0, noise_level_max=0.5):
    images_train = read_images(DATA_PATH_TRAIN)
    images_train_blur = read_images(DATA_PATH_TRAIN)
    print(images_train.shape)
    #print(type(images_train))

    labels_train = read_labels(LABEL_PATH_TRAIN)
    #print(labels_train.shape)

    images_test = read_images(DATA_PATH_TEST)
    #print(images_train.shape)

    labels_test = read_labels(LABEL_PATH_TEST)
    #print(labels_test.shape)

    images_train = images_train.astype(np.float32)
    images_train_blur = images_train_blur.astype(np.float32)

    #print("Max input:", np.max(images_train))
    #print("Min input:", np.min(images_train))

    images_train = normalize_min_max(images_train, 0.0, 1.0)
    #print("Max input:", np.max(images_train))
    #print("Min input:", np.min(images_train))
    #images_train_blur = images_train

    #images_train_blur = np.add(images_train, np.random.uniform(noise_level) * np.random.normal(loc=0, scale=(np.max(images_train) - np.min(images_train)) / 6., size=images_train.shape))
    for images in tqdm.tqdm(range(images_train_blur.shape[0]), total=images_train_blur.shape[0]):
        images_train_blur[images] = np.add(images_train_blur[images], np.random.uniform(noise_level_min, noise_level_max) * np.random.normal(loc=0, scale=(np.max(images_train_blur[images]) - np.min(images_train_blur[images])) / 6., size=images_train_blur[images].shape))
    images_train_blur = normalize_min_max(images_train_blur, 0.0, 1.0)
    #print(images_train_blur.shape)



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

    tb.add_text("Dataset", "Created dataset with: "
                           "\nGaussian noise between:" + str(noise_level_min) + " and " + str(noise_level_max) +
                           "\nBatch Size: " + str(batch_size) +
                           "\nTraining size: " + str(x_train.shape[0]) +
                           "\nValidation size:" + str(y_train.shape[0]))
    random_index = int(np.random.random()*len(images_train))-5
    tb.add_images("Dataset Examples - Original", images_train[random_index:random_index+5])
    #print("Max input:", np.max(images_train))
    #print("Min input:", np.min(images_train))
    tb.add_images("Dataset Examples - Blurred", images_train_blur[random_index:random_index+5])

    return train_Loader, val_Loader


def create_dataset_encrypt(batch_size = 8, key=128):
    images_train = read_images(DATA_PATH_TRAIN)
    print(images_train.shape)
    print(type(images_train))

    labels_train = read_labels(LABEL_PATH_TRAIN)
    print(labels_train.shape)

    images_test = read_images(DATA_PATH_TEST)
    print(images_train.shape)

    labels_test = read_labels(LABEL_PATH_TEST)
    print(labels_test.shape)

    # images_train_blur = create_blur_set(images_train, kx, ky)

    images_train = normalize_min_max(images_train, 0.0, 1.0)

    #images_train_blur = np.add(images_train, noise_level * np.random.normal(loc=0, scale=(np.max(images_train) - np.min(
    #    images_train)) / 6., size=images_train.shape))
    #images_train_blur = normalize_min_max(images_train_blur, 0.0, 1.0)
    #print(images_train_blur.shape)

    #images_train = images_train.astype(np.float32)
    #images_train_blur = images_train_blur.astype(np.float32)

    #(x_train, x_val, y_train, y_val) = train_test_split(images_train_blur, images_train, test_size=0.25)
    # print("Data shapes:")
    # print(f"x_train: {x_train.shape}")
    # print(f"x_val: {x_val.shape}")
    # print(f"y_train: {y_train.shape}")
    # print(f"y_val: {y_val.shape}")
    # print('Done shapes')

    #train_data = GaussDataset(x_train, y_train)
    #train_Loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    #val_data = GaussDataset(x_val, y_val)
    #val_Loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    #return train_Loader, val_Loader


def initialize_model(train=False, checkpoint='./checkpoints/latestModel.pth'):
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
    #summary(model, (1, 96, 96))

    if train:
        model.train()
    else:
        model.eval()
        load_checkpoint(checkpoint, model, optimizer)

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


def get_sample(model, dataset):
    model.eval()
    with torch.no_grad():
        data = iter(dataset).next()
        blur_img = data[0].to(device)
        sharp_img = data[1].to(device)
        output = model_RGB(model, blur_img)
    return blur_img[0], sharp_img[0], output[0]

def train(nr_epoch, model, trainLoader, valLoader, optimizer, criterion, lr_scheduler, tb):
    train_loss = []
    val_loss = []
    min_loss = sys.float_info.max
    start = time.time()
    now = datetime.now()
    today = datetime.today()
    today = today.strftime("%y_%m_%d")
    current_time = now.strftime("%H_%M_%S")
    last_ck = './checkpoints/Model_date_' + today + '_time_' + current_time + '.pth'
    print("Check . tensorboard --logdir=runs . for logs")
    tb.add_text("Training model start", "Model trained on:" + today + " at :" + current_time)

    for epoch in range(nr_epoch):
        logs = {}
        train_epoch_loss = train_step(model, trainLoader, optimizer, criterion, epoch)
        val_epoch_loss, samples = val_step(model, valLoader, optimizer, criterion, epoch)
        blur_img, sharp_img, output = get_sample(model, trainLoader)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        lr_scheduler.step(val_epoch_loss)


        #plot_loss(train_loss, val_loss)
        tb.add_scalar("Train Loss", train_epoch_loss, epoch)
        tb.add_scalar("Val Loss", val_epoch_loss, epoch)
        tb.add_image('Input - Gauss', blur_img, global_step=epoch)
        tb.add_image('Output', output, global_step=epoch)
        tb.add_image('Original', sharp_img, global_step=epoch)
        #print(samples.shape)
        #img_grid = torchvision.utils.make_grid(samples)
        #tb.add_image("Input - Output - Original", img_grid)

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
        print(f"################################ \n")
        print(f"Epoch Losses - Min Val: {min_loss:.5f} \n")
        print(f"Train: {train_epoch_loss:.5f} & Val: {val_epoch_loss:.5f} \n")
        save_checkpoint(last_ck, model, optimizer)
        print('Saved Checkpoint!')
        print(f"################################ \n")

    end = time.time()
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

def visual_test_tb(model, optimizer, checkpoint, dataLoader, maxImg):
    load_checkpoint(checkpoint, model, optimizer)
    model.eval()
    print('Visual Test TB!')
    print("Check . tensorboard --logdir=runs . for logs")
    tb = SummaryWriter()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size,
                                 position=0, leave=True):
            blur_img = data[0].to(device)
            sharp_img = data[1].to(device)
            output = model_RGB(model, blur_img)

            tb.add_images('Input', blur_img[0], global_step=i)
            tb.add_images('Output', sharp_img[0], global_step=i)
            tb.add_images('Original', output[0], global_step=i)
            input()
            print("Press return to serve new images")
            if i >= maxImg:
                break

    tb.close()

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
    nr_epoch = 25
    tb = SummaryWriter()
    #matplotlib.use('Agg')
    trainLoader, valLoader = create_dataset_noise(tb, batch_size=16, noise_level_min=0.1, noise_level_max=0.3)
    #check_pairs(trainLoader)
    checkpoint = './checkpoints/Model_20_26_38.pth'
    model, criterion, lr_scheduler, optimizer = initialize_model(train=True, checkpoint=checkpoint)
    model_summary = str(summary(model, input_size=(16, 1, 96, 96), verbose=0))
    tb.add_text("Model Summary", model_summary)
    train(nr_epoch, model, trainLoader, valLoader, optimizer, criterion, lr_scheduler, tb)
    #compare_gauss()

    #visual_test(model, optimizer, checkpoint, valLoader)
    #visual_test_tb(model, optimizer, checkpoint, valLoader, maxImg=150)

    #image += noise_level * torch.normal(mean=0, std=(image.max() - image.min()) / 6., size=image.shape).cuda()
    tb.close()
