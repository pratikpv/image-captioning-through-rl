###################################################
# Image Captioning with Deep Reinforcement Learning
# SJSU CMPE-297-03 | Spring 2020
#
#
# Team:
# Pratikkumar Prajapati
# Aashay Mokadam
# Karthik Munipalle
###################################################

import argparse
from datetime import datetime
from trainers import *
import warnings

# defaults and params
device = "cuda"
BASE_DIR = os.path.join('datasets', 'coco_captioning')  # path of the dataset
REAL_CAPTIONS_FILE = 'real_captions.txt'  # actual captions from the dataset stored in this file, for scoring
GENERATED_CAPTIONS_FILE = 'generated_captions.txt'  # generated captions are stored in this file, for scoring
IMAGE_URL_FILENAME = 'image_url.txt'  # actual image urls from the dataset stored in this file, for viewing results
LOG_DIR = ""  # all logs are save in this LOG_DIR, a value gets assigned based on execution date-time stamp

# network weight files
A2C_NETWORK_WEIGHTS_FILE = 'a2cNetwork.pt'
REWARD_NETWORK_WEIGHTS_FILE = 'rewardNetwork.pt'
POLICY_NETWORK_WEIGHTS_FILE = 'policyNetwork.pt'
VALUE_NETWORK_WEIGHTS_FILE = 'valueNetwork.pt'

RESULTS_FILE = 'results.txt'  # various scores are saved in this file
BEST_SCORE_FILENAME = 'best_scores.txt'  # post-processing stage saves best results in this file
BEST_SCORE_IMAGES_PATH = 'best_scores_images'  # # post-processing stage download images of best results here
# CURRICILUM_LEVELS = [2, 4, 6, 8, 10, 12, 14, 16]
CURRICILUM_LEVELS = [3, 6, 9, 12, 15]


def setup(args):
    """
    Create various configurations based on args.
    @param args: command line arguments
    @return: various dictionary objects with paths and configuration data
    """
    global LOG_DIR, device

    # execute on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print_green(f"[Info] Working on: {device}, device_name: {torch.cuda.get_device_name(0)} ")
    else:
        print_green(f"[Info] Working on: {device}")

    if os.path.isdir(os.path.split(args.test_model)[0]):
        LOG_DIR = os.path.split(args.test_model)[0]
    else:
        current_time_str = str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S"))
        LOG_DIR = os.path.join('logs', current_time_str)
        os.makedirs(LOG_DIR)

    warnings.filterwarnings("ignore", category=UserWarning)

    reward_file = get_filename(REWARD_NETWORK_WEIGHTS_FILE, args.bidirectional, None)
    policy_file = get_filename(POLICY_NETWORK_WEIGHTS_FILE, args.bidirectional, None)
    value_file = get_filename(VALUE_NETWORK_WEIGHTS_FILE, args.bidirectional, None)
    a2c_file = get_filename(A2C_NETWORK_WEIGHTS_FILE, args.bidirectional, args.curriculum)
    results_file = get_filename(RESULTS_FILE, args.bidirectional, args.curriculum)
    generated_captions_file = get_filename(GENERATED_CAPTIONS_FILE, args.bidirectional, args.curriculum)

    save_paths = {
        "model_path": os.path.join(LOG_DIR, a2c_file),
        "results_path": os.path.join(LOG_DIR, results_file),
    }

    image_caption_data = {
        "real_captions_path": os.path.join(LOG_DIR, REAL_CAPTIONS_FILE),
        "generated_captions_path": os.path.join(LOG_DIR, generated_captions_file),
        "image_urls_path": os.path.join(LOG_DIR, IMAGE_URL_FILENAME),
        "best_score_file_path": os.path.join(LOG_DIR, BEST_SCORE_FILENAME),
        "best_score_images_path": os.path.join(LOG_DIR, BEST_SCORE_IMAGES_PATH),
    }

    MODEL_DIRECTORY = args.pretrained_path
    network_paths = {
        "a2c_network": os.path.join(MODEL_DIRECTORY, a2c_file),
        "reward_network": os.path.join(MODEL_DIRECTORY, reward_file),
        "policy_network": os.path.join(MODEL_DIRECTORY, policy_file),
        "value_network": os.path.join(MODEL_DIRECTORY, value_file),
    }

    return save_paths, image_caption_data, network_paths


def main(args):
    """
    The main function to call various modules (train, test, post-process etc) based on args.
    @param args: command line arguments
    """
    save_paths, image_caption_data, network_paths = setup(args)

    print_green(f'[Info] Saving Logs in dir: {LOG_DIR}')

    max_train = None if args.training_size == 0 else args.training_size  # set None for whole training dataset
    max_train_str = '' if max_train == None else str(max_train)
    print_green(f'[Info] Loading COCO dataset {max_train_str}')
    data = load_data(base_dir=BASE_DIR, max_train=max_train, print_keys=True)
    print_green(f'[Info] COCO dataset loaded')

    train_corpus = None
    if args.train_word2vec != "none":
        print_green(f'[Info] Loading Word Embeddings {args.train_word2vec}')
        print_green(f'[Info] Loading Corpus')
        train_corpus = get_preprocessed_corpus(BASE_DIR)
        print_green(f'[Info] Corpus Loaded With {len(train_corpus)} Lines')
        data["embeddings"] = train_word_embeddings(args.train_word2vec, data, train_corpus)
        print_green(f'[Info] Done Loading Word Embeddings')
    else:
        data["embeddings"] = None

    if os.path.isfile(args.test_model) and "a2cNetwork" in os.path.split(args.test_model)[1]:
        print_green(f'[Info] Loading A2C Network')
        a2c_network = load_a2c_models(args.test_model, data, network_paths, args.bidirectional)
        print_green(f'[Info] A2C Network loaded')
    else:
        if args.curriculum:
            curriculum = CURRICILUM_LEVELS
        else:
            curriculum = None

        print_green(f'[Info] Training A2C Network')
        a2c_network = train_a2c_network(train_data=data, \
                                        save_paths=save_paths, network_paths=network_paths, \
                                        plot_dir=LOG_DIR, epochs=args.epochs, batch_size=args.batch_size, \
                                        bidirectional=args.bidirectional, retrain_all=args.retrain,
                                        curriculum=curriculum)
        print_green(f'[Info] A2C Network trained')

    print_green(f'[Info] Testing A2C Network')
    test_a2c_network(a2c_network, test_data=data, \
                     image_caption_data=image_caption_data, data_size=args.test_size)
    print_green(f'[Info] A2C Network Tested')

    print_green(f'[Info] A2C Network score - start')
    calculate_a2cNetwork_score(image_caption_data, save_paths)
    print_green(f'[Info] A2C Network score - end')

    if args.postprocess:
        print_green(f'[Info] Post-processing - start')
        post_process_data(image_caption_data)
        print_green(f'[Info] Post-processing - end')

    print_green(f'[Info] Logs saved in dir: {LOG_DIR}')


if __name__ == "__main__":
    # collect command line arguments for execution
    parser = argparse.ArgumentParser(description='Generate Image Captions through Deep Reinforcement Learning')

    parser.add_argument('--training_size', type=int, help='Size of the training set to use (set 0 for the full set)',
                        default=0)
    parser.add_argument('--test_size', type=int, help='Size of the test set to use', default=40504)

    parser.add_argument('--epochs', type=int, help='Number of Epochs to use for Training the A2C Network', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='Number of Episodes (Batch Size) to use for Training the A2C Network', default=512)

    parser.add_argument('--retrain', action='store_true', help='Whether to retrain value, policy and reward networks',
                        default=False)
    parser.add_argument('--postprocess', action='store_true',
                        help='Post process data to download images from the validation cycle', default=False)

    parser.add_argument('--curriculum', action='store_true', help='Use curriculum training approach', default=False)
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional recurrent neural networks',
                        default=False)

    parser.add_argument('--test_model', type=str, help='Test a pretrained advantage actor critic model', default="")
    parser.add_argument('--pretrained_path', type=str, help='Location of pretrained model files',
                        default="models_pretrained")

    # choices: ["none", "conceptnet", "word2vec", "fasttext", "glove", "path/to/word/embedding/model"]
    parser.add_argument('--pretrained_word2vec', type=str, help='Word Embedding model to use', default="none")
    parser.add_argument('--train_word2vec', type=str, choices=["none", "word2vec", "fasttext"],
                        help='Whether to train a word embedding model on training data', default="none")
    args = parser.parse_args()

    main(args)
