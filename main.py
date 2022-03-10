import binascii
import secrets

import livelossplot
import pbkdf2
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
from tqdm.auto import tqdm
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
import pyaes

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
DATA_PATH_TRAIN = './data/stl10_binary/unlabeled_X.bin'


def get_encrypted_dataset(num_images):
    """

        :param num_images: Number of images to retrieve, should be a multiple of 2000
        :return: numpy array with encrypted images
        """

    path = "./encDataset/EncryptedDataset__start_0_num_2000.npy"
    dataset = None

    if num_images <= 2000:
        print('Reading files from : %s', path)
        dataset = np.load(path)

    else:
        for i in range(int(num_images/2000)):
            path = "./encDataset/" + f"EncryptedDataset__start_{i*2000}_num_2000.npy"
            if dataset is None:
                dataset = np.load(path)
                print("Loaded", path)
            else:
                dataset = np.concatenate((dataset, np.load(path)), axis=0)
                print("Loaded", path)

    print(f"Loaded encrypted dataset of size {dataset.shape} \n Estimated size: {num_images * 0.421875} MB in RAM")
    return dataset


def read_images(path_to_data):
    print('Reading files from : %s', path_to_data)
    with open(path_to_data, 'rb') as f:
        input = np.fromfile(f, dtype=np.uint8)
        dataset_x = np.reshape(input, (-1, 3, 96, 96))
        return dataset_x


def normalize_min_max(x, new_min=0.0, new_max=1.0):
    min_val = np.min(x)
    #print("Min", min_val)
    max_val = np.max(x)
    #print("Max", max_val)
    x = (x - min_val) / (max_val - min_val)
    x = x * (new_max - new_min) + new_min
    #x = (x - np.min(x))/np.ptp(x)
    return x


#################### Model Helper Functions ##########################
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    #print('New checkpoint in:  %s' % checkpoint_path)


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
    def __init__(self, channel_in, channel_out, useBN=False):
        super(Net, self).__init__()

        self.conv1 = add_conv_stage(channel_in, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)
        self.conv5 = add_conv_stage(256, 512, useBN=useBN)

        self.conv4m = add_conv_stage(512, 256, useBN=useBN)
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128, 64, useBN=useBN)
        self.conv1m = add_conv_stage(64, 32, useBN=useBN)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, channel_out, 3, 1, 1),
            #nn.Sigmoid()
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

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

############################### DATASET CREATION #################################
def unpack_3d(model, img_in):
    dim3d = img_in.shape[1]
    img_2d = model(img_in[:, 0, :, :].unsqueeze(1))
    if dim3d > 1:
        for dim in range(dim3d-1):
            img_2d_new = model(img_in[:, dim+1, :, :].unsqueeze(1))
            img_2d = torch.cat((img_2d, img_2d_new), dim=1)
    img_out = torch.clamp(img_2d, 0., 1.)
    #print(img_out.shape)
    return img_out


class UnetDataset(Dataset):
    def __init__(self, input_paths, target_paths=None):
        self.X = input_paths
        self.y = target_paths

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        input_image = self.X[i, :, :, :]
        input_image = torch.from_numpy(input_image)

        if self.y is not None:
            target_image = self.y[i, :, :, :]
            target_image = torch.from_numpy(target_image)
            return input_image, target_image
        else:
            return input_image


def create_dataset_noise(tb, num_img = 10000, batch_size=8, noise_level_min=0.0, noise_level_max=0.5):
    images_train = read_images(DATA_PATH_TRAIN)[0:num_img]
    images_train_blur = read_images(DATA_PATH_TRAIN)[0:num_img]
    print(images_train.shape)
    #print(type(images_train))

    images_train = images_train.astype(np.float32)
    images_train_blur = images_train_blur.astype(np.float32)

    images_train = normalize_min_max(images_train, 0.0, 1.0)

    for images in tqdm(range(images_train_blur.shape[0]), total=images_train_blur.shape[0], ncols=100, desc="Creating Noise Dataset"):
        images_train_blur[images] = np.add(images_train_blur[images], np.random.uniform(noise_level_min, noise_level_max) * np.random.normal(loc=0, scale=(np.max(images_train_blur[images]) - np.min(images_train_blur[images])) / 6., size=images_train_blur[images].shape))
    images_train_blur = normalize_min_max(images_train_blur, 0.0, 1.0)

    (x_train, x_val, y_train, y_val) = train_test_split(images_train_blur, images_train, test_size=0.10)

    train_data = UnetDataset(x_train, y_train)
    train_Loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = UnetDataset(x_val, y_val)
    val_Loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    tb.add_text("Dataset", "Created dataset with: "
                           "\nGaussian noise between:" + str(noise_level_min) + " and " + str(noise_level_max) +
                           "\nBatch Size: " + str(batch_size) +
                           "\nTraining size: " + str(x_train.shape[0]) +
                           "\nValidation size:" + str(y_train.shape[0]))
    random_index = int(np.random.random()*len(images_train))-5
    tb.add_images("Dataset Examples - Original", images_train[random_index:random_index+5])
    tb.add_images("Dataset Examples - Blurred", images_train_blur[random_index:random_index+5])

    return train_Loader, val_Loader


def encrypt_img(image, key, iv):
    plaintext = image.tobytes()
    aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
    ciphertext = aes.encrypt(plaintext)
    #print("Cyp:", len(ciphertext))
    #print(ciphertext)
    encrypted_image = np.frombuffer(buffer=ciphertext, dtype=np.uint8)
    encrypted_image = encrypted_image.astype(np.float32)
    #print(ciphertext[0])
    #print(encrypted_image[0])
    #encrypted_image = np.nan_to_num(encrypted_image, copy=True, nan=0.0)
    #print("Min", np.min(encrypted_image))
    #print("Max", np.max(encrypted_image))
    #encrypted_image = normalize_min_max(encrypted_image, 0.0, 1.0)
    #print("Min",np.min(encrypted_image))
    #print("Max",np.max(encrypted_image))
    encrypted_image = encrypted_image.reshape((12, 96, 96))
    #print(encrypted_image)


    return encrypted_image


def create_dataset_encrypt(tb, num_images=10000, batch_size=8, key=128, iv=128, new=False, dataset_name="EncriptedDataset.npy"):

    images_train = read_images(DATA_PATH_TRAIN)[0:num_images]
    print(images_train.shape)
    print(type(images_train))

    images_train = images_train.astype(np.float32)

    images_train = normalize_min_max(images_train, 0.0, 1.0)

    if new:
        images_train_encrypt = read_images(DATA_PATH_TRAIN)[0:num_images]
        images_train_encrypt = images_train_encrypt.astype(np.float32)
        #images_train_encrypt = normalize_min_max(images_train_encrypt, 0.0, 1.0)

        temp = encrypt_img(images_train_encrypt[0], key, iv)
        temp = temp[np.newaxis, ...]
        for image in tqdm(range(images_train_encrypt.shape[0]-1), total=images_train_encrypt.shape[0]-1, desc="Creating Dataset Encrypt:"):
            result = encrypt_img(images_train_encrypt[image+1], key, iv)
            temp = np.concatenate((temp, result[np.newaxis, ...]))
            #print(temp.shape)
        images_train_encrypt = temp

        dataset_name = 'EncriptedDataset_' + model_type + '_date_' + today + '_time_' + current_time + "_size_" + str(num_images)

        np.save(dataset_name, images_train_encrypt)
        print("Saved dataset to:" + dataset_name)

        #images_train_encrypt = normalize_min_max(images_train_encrypt, 0.0, 1.0)

    else:
        print("Loading dataset from:" + dataset_name)
        images_train_encrypt = get_encrypted_dataset(num_images)
        print("Loaded dataset:", images_train_encrypt.shape)

    images_train_encrypt = normalize_min_max(images_train_encrypt, 0.0, 1.0)

    (x_train, x_val, y_train, y_val) = train_test_split(images_train_encrypt, images_train, test_size=0.10)

    train_data = UnetDataset(x_train, y_train)
    train_Loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = UnetDataset(x_val, y_val)
    val_Loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    random_index = int(np.random.random() * len(images_train)) - 5
    example = images_train_encrypt[random_index:random_index + 5, 0:3, :, :]
    print(example.shape)
    tb.add_text("Dataset Encrypted", "Created dataset with: "
                "\nKey:" + str(key) + " and IV " + str(iv) +
                "\nBatch Size: " + str(batch_size) +
                "\nTraining size: " + str(x_train.shape[0]) +
                "\nValidation size:" + str(y_train.shape[0]))
    tb.add_images("Dataset Examples - Original", images_train[random_index:random_index + 5])
    # print("Max input:", np.max(images_train))
    # print("Min input:", np.min(images_train))
    tb.add_images("Dataset Examples - Encrypted", example)

    return train_Loader, val_Loader


def initialize_model(train=False, ch_in=3, ch_out=3, checkpoint='./checkpoints/latestModel.pth'):
    model = Net(channel_in=ch_in, channel_out=ch_out).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)
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
    #print(' ')
    #print('Train Step:', epoch)
    for i, data in tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size, position=1, leave=False, desc="Train Step "+str(epoch), ncols=100):
    #for i, data in enumerate(dataLoader):
        in_img = data[0].to(device)
        sharp_img = data[1].to(device)
        optimizer.zero_grad()
        #output = unpack_3d(model, in_img)
        output = model(in_img)
        output = torch.clamp(output, min=0.0, max=1.0)
        loss = criterion(output, sharp_img)
        #print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(dataLoader.dataset)
    #print(f"Train Loss: {train_loss:.5f}")

    return train_loss


def val_step(model, dataLoader, criterion):
    model.eval()
    running_loss = 0.0
    #print(' ')
    #print('Eval Step')
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size, position=2, leave=False, ncols=100, desc="Validation Step:"):
        #for i, data in enumerate(dataLoader):
            in_img = data[0].to(device)
            sharp_img = data[1].to(device)
            #output = unpack_3d(model, in_img)
            output = model(in_img)
            output = torch.clamp(output, min=0.0, max=1.0)
            loss = criterion(output, sharp_img)
            running_loss += loss.item()
        val_loss = running_loss / len(dataLoader.dataset)
        #print(f"Val Loss: {val_loss:.5f}")



        return val_loss


def get_sample(model, dataset):
    model.eval()
    with torch.no_grad():
        data = iter(dataset).next()
        in_img = data[0].to(device)
        sharp_img = data[1].to(device)
        #output = unpack_3d(model, in_img)
        output = model(in_img)
        output = torch.clamp(output, min=0.0, max=1.0)
    return in_img[0, 0:3], sharp_img[0, 0:3], output[0, 0:3]


def train(nr_epoch, model, trainLoader, valLoader, optimizer, criterion, lr_scheduler, tb, model_type):
    train_loss = []
    val_loss = []
    min_loss = sys.float_info.max
    start = time.time()
    last_ck = './checkpoints/Model_ ' + model_type + '_date_' + today + '_time_' + current_time + '.pth'
    print("Check . tensorboard --logdir= noise_runs OR encrypted_runs. for logs")
    tb.add_text("Training model start", "Model " + model_type + " trained on:" + today + " at :" + current_time)

    for epoch in tqdm(range(nr_epoch), total=nr_epoch, desc="Training.. Epoch:", position=0, leave=True, ncols=100):
        train_epoch_loss = train_step(model, trainLoader, optimizer, criterion, epoch)
        val_epoch_loss = val_step(model, valLoader, criterion)
        in_img, sharp_img, output = get_sample(model, trainLoader)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        lr_scheduler.step(val_epoch_loss)

        tb.add_scalar("Train Loss", train_epoch_loss, epoch)
        tb.add_scalar("Val Loss", val_epoch_loss, epoch)
        tb.add_image('Input', in_img, global_step=epoch)
        tb.add_image('Output', output, global_step=epoch)
        tb.add_image('Original', sharp_img, global_step=epoch)
        #print(samples.shape)

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
            #print(f"################################ \n")
            #print(f"Epoch Losses - Min Val: {min_loss:.5f} \n")
            #print(f"Train: {train_epoch_loss:.5f} & Val: {val_epoch_loss:.5f} \n")
            save_checkpoint(last_ck, model, optimizer)
            #print('Saved Checkpoint!')
            #print(f"################################ \n")

    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes to train")


def visual_test_tb(tb, model, optimizer, checkpoint, dataLoader, maxImg):
    load_checkpoint(checkpoint, model, optimizer)
    model.eval()
    print('Visual Test TB!')
    print("Check . tensorboard --logdir=runs . for logs")
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataLoader), total=len(dataLoader.dataset) / dataLoader.batch_size,
                                 position=0, leave=True):
            in_img = data[0].to(device)
            sharp_img = data[1].to(device)
            #output = unpack_3d(model, in_img)
            output = model(in_img)

            tb.add_images('Input', in_img[0, 0:3], global_step=i)
            tb.add_images('Output', sharp_img[0, 0:3], global_step=i)
            tb.add_images('Original', output[0, 0:3], global_step=i)
            input()
            print("Press return to serve new images")
            if i >= maxImg:
                break

    tb.close()


def create_key():
    iv = secrets.randbits(128)

    password = "password"
    passwordSalt = os.urandom(16)
    print("Salt =", passwordSalt)

    key = pbkdf2.PBKDF2(password, passwordSalt).read(16)
    print("Returned Key = ", key)
    print('AES encryption key:', binascii.hexlify(key))

    print("Generated Key: ", key)
    print("Generated IV:", iv)

    return key, iv


if __name__ == '__main__':
    # Basic training parameters
    nr_epoch = 50
    batch_size = 32

    # Sets the number of images
    # WARNING: 25K Images start paging my PC 16 GB or RAM
    num_images = 10000

    # More like placeholders since they can't be uploaded to git :(
    checkpoint = './checkpoints/Model_20_26_38.pth'
    dataset_name = "EncriptedDataset.npy"

    # Don't chamge, 42GB and 20 hours went into this
    key = b']\xad\xf8,\xc6\x95\x86v\xd7\x11y\xf2\xc8\\n\xee'
    iv = 204520598923556323596093645128514940334

    # Parameters to switch between Noise and Encrypted inputs
    # Noise dataset is generated on the spot
    # Encryption dataset is saved on disk due to processing complexity of ground truth ~ 20 hours for entire dataset
    noise_level_min = 0.1
    noise_level_max = 0.3

    # Set model and dataset mode
    # True = Encrypted Dataset
    # False = Noisy Dataset
    encryption_mode = True

    # if the above is true, this sets if a new dataset should be created or loaded from disk
    new_dataset = False

    # Logging vars
    now = datetime.now()
    today = datetime.today()
    today = today.strftime("%y_%m_%d")
    current_time = now.strftime("%H_%M_%S")


    # Code for input switch
    # Inside, for Encryption mode, the dataset can be loaded if new=False, or can be created on the spot.
    # WARNING: Creating the encryption dataset takes a lot of time ~ 15 minutes per 2k images. See dataset_manipulation.py
    if encryption_mode:
        tb = SummaryWriter(log_dir="./encrypted_runs/m" + today + "_time_" + current_time)

        print("Encrpytion model")
        print("Check tensorboard ... --logdir=encrypted_runs ... in console")
        model_type = "Encrypt"

        if key is None or iv is None:
            print("Using generated key:")
            key, iv = create_key()

        tb.add_text("Encryption values", "Key=" + str(key) + "\nIV=" + str(iv))

        trainLoader, valLoader = create_dataset_encrypt(tb, num_images=num_images, batch_size=batch_size, key=key, iv=iv, new=new_dataset, dataset_name=dataset_name)

    else:
        tb = SummaryWriter(log_dir="./noise_runs/m" + today + "_time_" + current_time)
        print("Gauss model")
        print("Check tensorboard ... --logdir=noise_runs  ... in console")
        model_type = "Gauss"
        trainLoader, valLoader = create_dataset_noise(tb, num_img=num_images, batch_size=batch_size, noise_level_min=noise_level_min, noise_level_max=noise_level_max)

    sample1, sample2 = next(iter(trainLoader))
    ch_in = sample1.size(dim=1)
    ch_out = sample2.size(dim=1)
    print("Ch_in size:", ch_in)
    print("Ch_out size:", ch_out)

    # This initializes the model, nothing to change here, required for all the below functions
    model, criterion, lr_scheduler, optimizer = initialize_model(train=True, checkpoint=checkpoint, ch_in=ch_in, ch_out=ch_out)
    model_summary = str(summary(model, input_size=(batch_size, ch_in, 96, 96), verbose=0))
    tb.add_text("Model Summary", model_summary)

    # Uncomment this to train the model
    train(nr_epoch, model, trainLoader, valLoader, optimizer, criterion, lr_scheduler, tb, model_type=model_type)

    # Uncomment this to do a visual test in tensorboard
    # Must have a trained model and checkpoint saved for it
    #visual_test_tb(tb, model, optimizer, checkpoint="", valLoader, maxImg=150)

    #TO-DO Extra: Input transform from 12 to 3 (encrypt to RGB)

    tb.close()
