import time
import random
import torch.optim as optim
import os
from tqdm import tqdm
from utility_functions import *
from reinforcement_learning_networks import *
from torch.utils.tensorboard import SummaryWriter


def train_value_network(train_data, network_paths, plot_dir, batch_size=50, epochs=50000):

    value_writer = SummaryWriter(log_dir = os.path.join(plot_dir, 'runs'))

    rewardNet = RewardNetwork(train_data["word_to_idx"]).to(device)
    rewardNet.load_state_dict(torch.load(network_paths["reward_network"], map_location=device))
    for param in rewardNet.parameters():
        param.require_grad = False
    print(rewardNet)

    policyNet = PolicyNetwork(train_data["word_to_idx"]).to(device)
    policyNet.load_state_dict(torch.load(network_paths["policy_network"], map_location=device))
    for param in policyNet.parameters():
        param.require_grad = False
    print(policyNet)

    valueNetwork = ValueNetwork(train_data["word_to_idx"]).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(valueNetwork.parameters(), lr=0.0001)
    valueNetwork.train(mode=True)

    bestLoss = 10000
    max_seq_len = 17

    print(f'[Info] Training Value Network\n')
    for epoch in range(epochs):
        captions, features, _ = sample_coco_minibatch(train_data, batch_size=batch_size, split='train')
        features = torch.tensor(features, device=device).float()
        
        # Generate captions using the policy network
        # captions = GenerateCaptions(features, captions, policyNet)

        # Generate Captions using policy and value networks (Look Ahead Inference)
        captions = GenerateCaptionsLI(features, captions, policyNet, valueNetwork)
        
        # Compute the reward of the generated caption using reward network
        rewards = GetRewards(features, captions, rewardNet)
        
        # Compute the value of a random state in the generation process
    #     print(features.shape, captions[:, :random.randint(1, 17)].shape)
        values = valueNetwork(features, captions[:, :random.randint(1, 17)])
        
        # Compute the loss for the value and the reward
        loss = criterion(values, rewards)
        
        if loss.item() < bestLoss:
            bestLoss = loss.item()
            torch.save(valueNetwork.state_dict(), network_paths["value_network"])
            
            print("epoch:", epoch, "loss:", loss.item())

        value_writer.add_scalar('Value Network',loss,epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        valueNetwork.valrnn.hidden_cell[0].detach_()
        valueNetwork.valrnn.hidden_cell[1].detach_()
        rewardNet.rewrnn.hidden_cell.detach_()
    
    return valueNetwork


def train_policy_network(train_data, network_paths, plot_dir, batch_size=100, epochs=100000, pretrained=False):

    policyNetwork = PolicyNetwork(train_data["word_to_idx"]).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(policyNetwork.parameters(), lr=0.0001)

    policy_writer = SummaryWriter(log_dir = os.path.join(plot_dir, 'runs'))

    if pretrained:
        policyNetwork.load_state_dict(torch.load(network_paths["policy_network"], map_location=device))  
    
    bestLoss = 1.0
    print(f'[Info] Training Policy Network\n')
    for epoch in range(epochs):
        captions, features, _ = sample_coco_minibatch(train_data, batch_size=batch_size, split='train')
        features = torch.tensor(features, device=device).float().unsqueeze(0)
        captions_in = torch.tensor(captions[:, :-1], device=device).long()
        captions_out = torch.tensor(captions[:, 1:], device=device).long()
        output = policyNetwork(features, captions_in)
        
        loss = 0
        for i in range(batch_size):
            caplen = np.nonzero(captions[i] == 2)[0][0] + 1
            loss += (caplen/batch_size)*criterion(output[i][:caplen], captions_out[i][:caplen])
        
        if loss.item() < bestLoss:
            bestLoss = loss.item()
            torch.save(policyNetwork.state_dict(), network_paths["policy_network"])
            
            print("epoch:", epoch, "loss:", loss.item())

        policy_writer.add_scalar('Policy Network',loss,epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_reward_network(train_data, network_paths, plot_dir, batch_size=50, epochs=50000):

    reward_writer = SummaryWriter(log_dir = os.path.join(plot_dir, 'runs'))
    rewardNetwork = RewardNetwork(train_data["word_to_idx"]).to(device)
    optimizer = optim.Adam(rewardNetwork.parameters(), lr=0.001)  

    bestLoss = 10000
    print(f'[Info] Training Reward Network\n')
    for epoch in range(epochs):
        captions, features, _ = sample_coco_minibatch(train_data, batch_size=batch_size, split='train')
        features = torch.tensor(features, device=device).float()
        captions = torch.tensor(captions, device=device).long()
        ve, se = rewardNetwork(features, captions)
        loss = VisualSemanticEmbeddingLoss(ve, se)
        
        if loss.item() < bestLoss:
            bestLoss = loss.item()
            torch.save(rewardNetwork.state_dict(), network_paths["reward_network"])
            
            print("epoch:", epoch, "loss:", loss.item())

        reward_writer.add_scalar('Reward Network',loss,epoch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewardNetwork.rewrnn.hidden_cell.detach_()

    return rewardNetwork


def train_a2c_network(train_data, save_paths, network_paths, plot_dir, epoch_count=10, episodes=100, usePretrained=True, plot_freq=10):
    
    a2c_train_writer = SummaryWriter(log_dir=os.path.join(plot_dir,'runs'))

    model_save_path = save_paths["model_path"]
    results_save_path = save_paths["results_path"]

    if usePretrained:
        rewardNet = RewardNetwork(train_data["word_to_idx"]).to(device)
        policyNet = PolicyNetwork(train_data["word_to_idx"]).to(device)
        valueNet = ValueNetwork(train_data["word_to_idx"]).to(device)

        rewardNet.load_state_dict(torch.load(network_paths["reward_network"], map_location=device))
        policyNet.load_state_dict(torch.load(network_paths["policy_network"], map_location=device))
        valueNet.load_state_dict(torch.load(network_paths["value_network"], map_location=device))

    else:
        rewardNet = train_reward_network(train_data, network_paths)
        policyNet = train_policy_network(train_data, network_paths)
        valueNet = train_value_network(train_data, network_paths)

    # rewardNet.train(mode=False)

    a2cNetwork = AdvantageActorCriticNetwork(valueNet, policyNet).to(device)
    a2cNetwork.train(True)
    optimizer = optim.Adam(a2cNetwork.parameters(), lr=0.0001)

    print(f'[training] train_data len = {len(train_data["train_captions"])}')
    print(f'[training] episodes = {episodes}')
    print(f'[training] epoch_count = {epoch_count}')

    for epoch in range(epoch_count):
        episodicAvgLoss = 0

        captions, features, _ = sample_coco_minibatch(train_data, batch_size=episodes, split='train')
        features = torch.tensor(features, device=device).float()
        captions = torch.tensor(captions, device=device).long()

        episode_t = time.time()
        for episode in range(episodes):

            captions_in = captions[episode:episode + 1, :]
            features_in = features[episode:episode + 1]

            values, probs = a2cNetwork(features_in, captions_in)

            probs = F.softmax(probs, dim=2)
            dist = probs.cpu().detach().numpy()[0, 0]
            action = np.random.choice(probs.shape[-1], p=dist)

            gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
            try:
                captions_in = torch.cat((captions_in, gen_cap), axis=1)
            except:
                captions_in = torch.cat((captions_in, gen_cap.long()), axis=1)

            log_probs = torch.log(probs[0, 0, action])

            rewards = GetRewards(features_in, captions_in, rewardNet)
            rewards = rewards.cpu().detach().numpy()[0, 0]

            values = torch.FloatTensor([values]).to(device)
            rewards = torch.FloatTensor([rewards]).to(device)
            log_probs = torch.stack([log_probs]).to(device)

            advantage = values - rewards
            actorLoss = (-log_probs * advantage).mean()
            criticLoss = 0.5 * advantage.pow(2).mean()

            loss = actorLoss + criticLoss
            episodicAvgLoss += loss.item() / episodes

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if episode % 1000 == 0:
                print("[training] episode: %s, time taken: %ss" % (episode, time.time() - episode_t))
                print("[training] current memory allocated: %s\t | cached memory: %s" \
                                % (torch.cuda.memory_allocated() / 1024 ** 2, \
                                        torch.cuda.memory_cached() / 1024 ** 2))
                # print_garbage_collection()
                episode_t = time.time()

            ## Summary Writer
            if episode % plot_freq == 0:
                a2c_train_writer.add_scalar('A2C Network',episodicAvgLoss,episode)
        
        print(f"[training] epoch:{epoch} episodicAvgLoss: {episodicAvgLoss}")
        rewardNet.rewrnn.init_hidden()
        valueNet.valrnn.init_hidden()

    torch.save(a2cNetwork.state_dict(), model_save_path)
    with open(results_save_path, 'a') as f:
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')
        f.write(str(a2cNetwork))
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')

    return a2cNetwork


def test_a2c_network(a2cNetwork, test_data, image_caption_data, data_size, validation_batch_size=100):

    a2c_test_writer = SummaryWriter()
    a2cNetwork.train(False)

    real_captions_filename = image_caption_data["real_captions_path"]
    generated_captions_filename = image_caption_data["generated_captions_path"]
    image_url_filename = image_caption_data["image_urls_path"]

    real_captions_file = open(real_captions_filename, "a")
    generated_captions_file = open(generated_captions_filename, "a")
    image_url_file = open(image_url_filename, "a")

    captions_real_all, features_real_all, urls_all = sample_coco_minibatch(test_data, batch_size=data_size, split='val')
    val_captions_lens = len(captions_real_all)
    loop_count = val_captions_lens // validation_batch_size

    for i in tqdm(range(loop_count), desc='Testing model'):
        captions_real = captions_real_all[i:i + validation_batch_size - 1]
        features_real = features_real_all[i:i + validation_batch_size - 1]
        urls = urls_all[i:i + validation_batch_size]

        for j in range(validation_batch_size - 1):
            captions_real_v = captions_real[j:j+1]
            features_real_v = features_real[j:j+1]

            # value, probs = a2cNetwork(features_real_v, captions_real_v)
            # probs = F.softmax(probs, dim=2)
            # dist = probs.cpu().detach().numpy()[0, 0]
            # action = np.random.choice(probs.shape[-1], p=dist)
            # gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
            # gen_cap_str = decode_captions(gen_cap, idx_to_word=test_data["idx_to_word"])[0]

            gen_cap = GenerateCaptionsLI(features_real_v, captions_real_v, a2cNetwork.policyNet, a2cNetwork.valueNet, most_likely=True)[0]
            gen_cap_str = decode_captions(gen_cap, idx_to_word=test_data["idx_to_word"])
            real_cap_str = decode_captions(captions_real[j], idx_to_word=test_data["idx_to_word"])

            real_captions_file.write(real_cap_str + '\n')
            generated_captions_file.write(gen_cap_str + '\n')
            image_url_file.write(urls[j] + '\n')

            real_captions_file.flush()
            generated_captions_file.flush()
            image_url_file.flush()

    real_captions_file.close()
    generated_captions_file.close()
    image_url_file.close()


def load_a2c_models(model_path, train_data, network_paths):
    
    policyNet = PolicyNetwork(train_data["word_to_idx"]).to(device)
    policyNet.load_state_dict(torch.load(network_paths["policy_network"], map_location=device))
    policyNet.train(mode=False)

    valueNet = ValueNetwork(train_data["word_to_idx"]).to(device)
    valueNet.load_state_dict(torch.load(network_paths["value_network"], map_location=device))
    valueNet.train(mode=False)

    a2cNetwork = AdvantageActorCriticNetwork(valueNet, policyNet).to(device)
    a2cNetwork.load_state_dict(torch.load(model_path, map_location=device))

    return a2cNetwork
