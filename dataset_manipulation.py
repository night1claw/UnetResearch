import numpy as np
from tqdm.auto import tqdm
import pyaes
from datetime import datetime

DATA_PATH_TRAIN = './data/stl10_binary/unlabeled_X.bin'


def read_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        input = np.fromfile(f, dtype=np.uint8)

        dataset_x = np.reshape(input, (-1, 3, 96, 96))
        return dataset_x


def encrypt_img(image, key, iv):
    plaintext = image.tobytes()
    aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
    ciphertext = aes.encrypt(plaintext)
    encrypted_image = np.frombuffer(buffer=ciphertext, dtype=np.uint8)
    encrypted_image = encrypted_image.astype(np.float32)
    encrypted_image = encrypted_image.reshape((1, 12, 96, 96))

    return encrypted_image


def create_dataset_encrypt(start=0, num_images=10000, key=b']\xad\xf8,\xc6\x95\x86v\xd7\x11y\xf2\xc8\\n\xee', iv=128):
    images_train_encrypt = read_images(DATA_PATH_TRAIN)[start:start+num_images]
    images_train_encrypt = images_train_encrypt.astype(np.float32)
    # images_train_encrypt = normalize_min_max(images_train_encrypt, 0.0, 1.0)

    temp = encrypt_img(images_train_encrypt[0], key, iv)
    for image in tqdm(range(images_train_encrypt.shape[0] - 1), total=images_train_encrypt.shape[0] - 1,
                      desc="Creating Dataset Encrypt:" + str(start) + " to " + str(start+num_images), position=1, leave=False, ncols=120):
        result = encrypt_img(images_train_encrypt[image + 1], key, iv)
        temp = np.concatenate((temp, result))
        #print(temp.shape)
    images_train_encrypt = temp
    dataset_name = './encDataset/EncryptedDataset_' + "_start_" + str(start) + "_num_" + str(num_images)

    np.save(dataset_name, images_train_encrypt)
    print("Saved dataset to:" + dataset_name)

    # images_train_encrypt = normalize_min_max(images_train_encrypt, 0.0, 1.0)


if __name__ == '__main__':
    dataset_name = "EncriptedDataset.npy"
    num_images = 2000
    splits = 50
    if not 100000 % num_images:
        print("Not divisible!")
    else:
        splits = 100000 / num_images
        splits = int(splits)

    key = b']\xad\xf8,\xc6\x95\x86v\xd7\x11y\xf2\xc8\\n\xee'
    iv = 204520598923556323596093645128514940334

    now = datetime.now()
    today = datetime.today()
    today = today.strftime("%y_%m_%d")
    current_time = now.strftime("%H_%M_%S")
    print("Num of splits = ", splits)
    print("Started Creating dataset at : ", now)
    for i in tqdm(range(splits), total=splits, desc="Encrypting splits:", position=0, leave=True, ncols=120):
        create_dataset_encrypt(start=i*num_images, num_images=num_images, key=key, iv=iv)
    print("Finished Creating dataset at : ", now)
