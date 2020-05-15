import h5py
import json
import requests
import gc
import os

from PIL import Image
from io import BytesIO
import urllib.request

import numpy as np
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from tqdm import tqdm
from models import *
from metrics import *


def print_green(text):
    print('\033[32m', text, '\033[0m', sep='')


def print_red(text):
    print('\033[31m', text, '\033[0m', sep='')


def load_data(base_dir, max_train=None, pca_features=True, print_keys=False):
    data = {}

    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    data["train_captions_lens"] = np.zeros(data["train_captions"].shape[0])
    data["val_captions_lens"] = np.zeros(data["val_captions"].shape[0])
    for i in range(data["train_captions"].shape[0]):
        data["train_captions_lens"][i] = np.nonzero(data["train_captions"][i] == 2)[0][0] + 1
    for i in range(data["val_captions"].shape[0]):
        data["val_captions_lens"][i] = np.nonzero(data["val_captions"][i] == 2)[0][0] + 1

    if print_keys:
        # Print out all the keys and values from the data dictionary
        for k, v in data.items():
            if type(v) == np.ndarray:
                print(k, type(v), v.shape, v.dtype)
            else:
                print(k, type(v), len(v))

    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def get_coco_batch(data, batch_size=100, split='train'):
    split_total_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_total_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls


def get_coco_minibatches(data, batch_size=100, split='train'):

    from torch import randperm
    split_total_size = data['%s_captions' % split].shape[0]
    permutation = torch.randperm(split_total_size)

    for i in range(0, split_total_size, batch_size):
        mask = permutation[i: i+batch_size]
        captions = data['%s_captions' % split][mask]
        image_idxs = data['%s_image_idxs' % split][mask]
        image_features = data['%s_features' % split][image_idxs]
        urls = data['%s_urls' % split][image_idxs]

        yield captions, image_features, urls


def get_coco_validation_data(data, data_size=None):
    captions = data['val_captions']
    image_features = data['val_features']
    urls = data['val_urls']
    return captions, image_features, urls


def image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def global_minibatch_number(epoch, batch_id, batch_size):
    return epoch*batch_size + batch_id

def print_garbage_collection():
    print("-" * 30)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
    print("-" * 30)


def post_process_data(image_caption_data, top_item_count=5):
    score_list = []

    real_captions_filename = image_caption_data["real_captions_path"]
    generated_captions_filename = image_caption_data["generated_captions_path"]
    image_url_filename = image_caption_data["image_urls_path"]
    best_score_file_path = image_caption_data["best_score_file_path"]
    best_score_images_path = image_caption_data["best_score_images_path"]

    real_captions_file = open(real_captions_filename, "r")
    generated_captions_file = open(generated_captions_filename, "r")
    image_url_file = open(image_url_filename, "r")
    best_score_file = open(best_score_file_path, "w")

    real_captions_lines = real_captions_file.readlines()
    generated_captions_lines = generated_captions_file.readlines()
    image_url_lines = image_url_file.readlines()
    data_len = len(real_captions_lines)

    for i in tqdm(range(data_len), desc='Comparing scores'):
        s = get_singleton_score(real_captions_lines[i], generated_captions_lines[i])
        avg = 0.0
        for k in s.keys():
            avg += s[k]
        avg /= len(s.keys())
        score_list.append(avg)

    arr = np.array(score_list)
    top_items_index = arr.argsort()[::-1][:top_item_count]

    if not os.path.isdir(best_score_images_path):
        os.mkdir(best_score_images_path)

    for i in tqdm(top_items_index, desc='Downloading images'):
        buff = 'item_index[%d] score:[%f] real_cap:[%s] generated_cap:[%s] \n' % (
            i + 1, score_list[i], real_captions_lines[i].strip(), generated_captions_lines[i].strip())
        best_score_file.write(buff)
        try:
            i_name = "%d.jpg" % (i + 1)
            i_name = str(os.path.join(best_score_images_path, i_name))
            urllib.request.urlretrieve(image_url_lines[i], i_name)
        except:
            e = sys.exc_info()[0]
            print(f'downloading {image_url_lines[i]} failed with {e}')

    real_captions_file.close()
    generated_captions_file.close()
    image_url_file.close()
    best_score_file.close()


def save_a2c_model(model, save_paths):
    from torch import save

    if isinstance(save_paths, list):
        for path in save_paths:
            save(model.state_dict(), path)
    elif isinstance(save_paths, str):
        save(model.state_dict(), save_paths)


def load_a2c_models(model_path, train_data, network_paths, bidirectional):
    
    policy_network = PolicyNetwork(train_data["word_to_idx"], \
                        pretrained_embeddings=train_data["embeddings"], \
                            bidirectional=bidirectional).to(device)
    policy_network.load_state_dict(torch.load(network_paths["policy_network"], map_location=device), strict=False)
    policy_network.train(mode=False)

    value_network = ValueNetwork(train_data["word_to_idx"], \
                        pretrained_embeddings=train_data["embeddings"], \
                            bidirectional=bidirectional).to(device)
    value_network.load_state_dict(torch.load(network_paths["value_network"], map_location=device), strict=False)
    value_network.train(mode=False)

    a2c_network = AdvantageActorCriticNetwork(value_network, policy_network).to(device)
    a2c_network.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    return a2c_network

def get_filename(base_name, bidirectional, curriculum=None):

    name, ext = os.path.splitext(base_name)
    if bidirectional:
        name += "_bidirectional"
    if curriculum is not None:
        if curriculum:
            name += "_curriculum"
    name += ext

    return name


def calculate_a2cNetwork_score(image_caption_data, save_paths):
    
    real_captions_filename = image_caption_data["real_captions_path"]
    generated_captions_filename = image_caption_data["generated_captions_path"]

    ref, hypo = load_textfiles(real_captions_filename, generated_captions_filename)
    network_score = str(score(ref, hypo))
    print(network_score)

    results_filename = os.path.join(save_paths["results_path"])
    with open(results_filename, 'a') as f:
        f.write('\n' + '-' * 10 + ' results ' + '-' * 10 + '\n')
        f.write(network_score)
        f.write('\n' + '-' * 10 + ' results ' + '-' * 10 + '\n')


def get_preprocessed_corpus(base_dir):

    data = load_data(base_dir=base_dir, max_train=None, print_keys=False)
    idx_to_word = data["idx_to_word"]
    corpus_data = [simple_preprocess(" ".join([idx_to_word[d] for d in sent])) for sent in data["train_captions"]]
    corpus_data += [simple_preprocess(" ".join([idx_to_word[d] for d in sent])) for sent in data["val_captions"]]

    return corpus_data

def get_embeddings(emb_type):
    
    embeddings = None
    emb_name = ""

    if emb_type == "conceptnet":
        emb_name = "conceptnet-numberbatch-17-06-300"
    elif emb_type == "fasttext":
        emb_name = "fasttext-wiki-news-subwords-300"
    elif emb_type == "word2vec":
        emb_name = "word2vec-google-news-300"
    elif emb_type == "glove":
        emb_name = "glove-wiki-gigaword-300"
    elif os.path.isfile(emb_name):
        return emb_name
    
    embeddings = api.load(emb_name)

    return embeddings


def get_embedding_model(path):

    if isinstance(path, gensim.models.keyedvectors.BaseKeyedVectors):
        model = path
    elif isinstance(path, gensim.models.base_any2vec.BaseWordEmbeddingsModel):
        model = path.wv
    elif os.path.isfile(path):
        model = KeyedVectors.load_word2vec_format(path)
    else:
        raise ValueError("Got %s as an argument, expect either a path to embeddings or an embedding model" % type(path))

    return model


def get_vectors_by_by_vocab(model, word_to_idx):

    idx_to_word = {i: w for w, i in word_to_idx.items()}
    new_vectors = np.empty((len(idx_to_word), model.vectors.shape[1]), dtype=np.float32)
    curr_vecs = []

    for idx, word in idx_to_word.items():
        try:
            new_vectors[idx] = model[word]
            curr_vecs.append(model[word])
        except KeyError:
            # Initialize randomly
            if len(curr_vecs) == 0:
                new_vectors[idx] = np.random.rand(1, model.vectors.shape[1])
            # Initialize with mean of all vectors
            else:
                new_vectors[idx] = np.array(curr_vecs).mean(axis=0)

    return new_vectors


def load_word_embeddings(embedding_type, target_data, train_corpus=None):

    if embedding_type == "none":
        return None

    if embedding_type == "fasttext":
        model = gensim.models.FastText(sg=1, size=300, min_count=1, workers=56, word_ngrams=1)
    else:
        model = gensim.models.Word2Vec(sg=1, size=300, min_count=1, workers=56)

    model.build_vocab(train_corpus)

    print_green(f'[Info] Training Word Embeddings')
    model.train(train_corpus, total_examples=model.corpus_count, epochs=30, report_delay=5)
    print_green(f'[Info] Finished Training Word Embeddings')

    vectors = get_vectors_by_by_vocab(model.wv, target_data["word_to_idx"])

    return vectors
