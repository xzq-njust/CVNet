import os
import glob
import random
from package.config import config

def enumerate_train_test(filenames, num_train):
    all_train_examples = []
    all_test_examples = []

    # DSAC does not shuffle filenames, just cut them off at the right numbers
    for i, name in enumerate(filenames):
        if i < num_train:
            all_train_examples.append(name)
        else:
            all_test_examples.append(name)

    return all_train_examples, all_test_examples


def enumerate_train_val_test(filenames, num_img):
    random.shuffle(filenames)

    num_train = int(num_img*0.6)
    num_val = int(num_img*0.2)

    all_train_examples = []
    all_val_examples = []
    all_test_examples = []

    # DSAC does not shuffle filenames, just cut them off at the right numbers
    for i, name in enumerate(filenames):
        if i < num_train:
            all_train_examples.append(name)
        elif i < (num_train+num_val):
            all_val_examples.append(name)
        else:
            all_test_examples.append(name)

    return all_train_examples, all_val_examples, all_test_examples


def enumerate_train_val(filenames, num_img):
    random.shuffle(filenames)

    num_train = int(num_img*0.8)
    num_val = int(num_img*0.2)

    train_examples = []
    val_examples = []

    # DSAC does not shuffle filenames, just cut them off at the right numbers
    for i, name in enumerate(filenames):
        if i < num_train:
            train_examples.append(name)
        else:
            val_examples.append(name)

    train_examples.sort()
    val_examples.sort()

    return train_examples, val_examples


# split vaihingen
def split_vaihingen(num_train=100):
    cfg = config['vaihingen']
    data_path = cfg['data_path']
    image_glob = os.path.join(data_path, 'building_[0-9]*.tif')
    image_paths = glob.glob(image_glob)
    train_examples, test_examples = enumerate_train_test(sorted(image_paths), num_train)
    pretrain_examples, preval_examples = enumerate_train_val(sorted(train_examples), num_train)

    train_file = cfg['train']
    test_file = cfg['test']

    pretrain_file = config['pretrain_vaihingen']['train']
    preval_file = config['pretrain_vaihingen']['test']

    with open(train_file, 'w') as file:
        for name in train_examples:
            file.write('{}\n'.format(name))

    with open(test_file, 'w') as file:
        for name in test_examples:
            file.write('{}\n'.format(name))

    with open(pretrain_file, 'w') as file:
        for name in pretrain_examples:
            file.write('{}\n'.format(name))

    with open(preval_file, 'w') as file:
        for name in preval_examples:
            file.write('{}\n'.format(name))

# split bing
def split_bing(num_train=335):
    cfg = config['bing']
    data_path = cfg['data_path']
    image_glob = os.path.join(data_path, 'building_[0-9]*.png')
    image_paths = glob.glob(image_glob)
    train_examples, test_examples = enumerate_train_test(sorted(image_paths), num_train)
    pretrain_examples, preval_examples = enumerate_train_val(sorted(train_examples), num_train)

    train_file = cfg['train']
    test_file = cfg['test']

    pretrain_file = config['pretrain_bing']['train']
    preval_file = config['pretrain_bing']['test']

    with open(train_file, 'w') as file:
        for name in train_examples:
            file.write('{}\n'.format(name))

    with open(test_file, 'w') as file:
        for name in test_examples:
            file.write('{}\n'.format(name))

    with open(pretrain_file, 'w') as file:
        for name in pretrain_examples:
            file.write('{}\n'.format(name))

    with open(preval_file, 'w') as file:
        for name in preval_examples:
            file.write('{}\n'.format(name))


def split_inria():
    cfg = config['inria']
    data_path = cfg['img_path']
    image_glob = os.path.join(data_path, '*.tif')
    image_paths = glob.glob(image_glob)
    image_paths = [os.path.basename(image_path) for image_path in image_paths]
    
    num_img = len(image_paths)
    train_examples, val_examples, test_examples = enumerate_train_val_test(image_paths, num_img)

    train_file = cfg['train']
    val_file = cfg['val']
    test_file = cfg['test']

    with open(train_file, 'w') as file:
        for name in train_examples:
            file.write('{}\n'.format(name))

    with open(val_file, 'w') as file:
        for name in val_examples:
            file.write('{}\n'.format(name))

    with open(test_file, 'w') as file:
        for name in test_examples:
            file.write('{}\n'.format(name))



if __name__ == "__main__":
    split_vaihingen()
    split_bing()
    split_inria()
