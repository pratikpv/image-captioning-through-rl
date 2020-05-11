import time
import random
import torch.optim as optim
from tqdm import tqdm
from utilities import *
from models import *
from torch.utils.tensorboard import SummaryWriter


# https://cs230-stanford.github.io/pytorch-nlp.html#writing-a-custom-loss-function
def VisualSemanticEmbeddingLoss(visuals, semantics):
    beta = 0.2
    N, D = visuals.shape
    
    visloss = torch.mm(visuals, semantics.t())
    visloss = visloss - torch.diag(visloss).unsqueeze(1)
    visloss = visloss + (beta/N)*(torch.ones((N, N)).to(device) - torch.eye(N).to(device))
    visloss = F.relu(visloss)
    visloss = torch.sum(visloss)/N
    
    semloss = torch.mm(semantics, visuals.t())
    semloss = semloss - torch.diag(semloss).unsqueeze(1)
    semloss = semloss + (beta/N)*(torch.ones((N, N)).to(device) - torch.eye(N).to(device))
    semloss = F.relu(semloss)
    semloss = torch.sum(semloss)/N
    
    return visloss + semloss

def GenerateCaptions(features, captions, policy_network):
    features = torch.tensor(features, device=device).float().unsqueeze(0)
    gen_caps = torch.tensor(captions[:, 0:1], device=device).long()
    for t in range(MAX_SEQ_LEN - 1):
        output = policy_network(features, gen_caps)
        gen_caps = torch.cat((gen_caps, output[:, -1:, :].argmax(axis=2)), axis=1)
    return gen_caps

def GenerateCaptionsWithActorCriticLookAhead(features, captions, policy_network, value_network, beamSize=5, most_likely=False):

    features = torch.tensor(features, device=device).float().unsqueeze(0)
    gen_caps = torch.tensor(captions[:, 0:1], device=device).long()
    
    candidates = [(gen_caps, 0)]
    for t in range(MAX_SEQ_LEN-1):
        next_candidates = []
        for c in range(len(candidates)):
            output = policy_network(features, candidates[c][0])
            probs, words = torch.topk(output[:,-1:,:], beamSize)
            for i in range(beamSize):
                cap = torch.cat((candidates[c][0], words[:, :, i]), axis=1)
                value = value_network(features.squeeze(0), cap).detach()
                score_delta = 0.6*value + 0.4*torch.log(probs[:,:,i])
                score = candidates[c][1] - score_delta
                next_candidates.append((cap, score))
        ordered_candidates = sorted(next_candidates, key=lambda tup:tup[1].mean())
        candidates = ordered_candidates[:beamSize]
    
    if most_likely == True:
        return candidates[0][0]
    return candidates


def GetRewards(features, captions, reward_network):

    visEmbeds, semEmbeds = reward_network(features, captions)
    visEmbeds = F.normalize(visEmbeds, p=2, dim=1)
    semEmbeds = F.normalize(semEmbeds, p=2, dim=1)

    rewards = torch.sum(visEmbeds * semEmbeds, axis=1).unsqueeze(1)
    return rewards


def train_value_network(train_data, network_paths, plot_dir, batch_size=256, epochs=50000):

    value_writer = SummaryWriter(log_dir = os.path.join(plot_dir, 'runs'))

    reward_network = RewardNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
    reward_network.load_state_dict(torch.load(network_paths["reward_network"], map_location=device))
    for param in reward_network.parameters():
        param.require_grad = False


    policy_network = PolicyNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
    policy_network.load_state_dict(torch.load(network_paths["policy_network"], map_location=device))
    for param in policy_network.parameters():
        param.require_grad = False


    value_network = ValueNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(value_network.parameters(), lr=0.0001)
    value_network.train(mode=True)

    best_loss = float('inf')
    print_green(f'[Training] Training Value Network')

    progress = tqdm(range(epochs), desc='Training Value Network: Best Loss %s' % best_loss)
    for epoch in progress:
        captions, features, _ = get_coco_batch(train_data, batch_size=batch_size, split='train')
        features = torch.tensor(features, device=device).float()

        # Generate captions using the policy network
        captions = GenerateCaptions(features, captions, policy_network)

        # Compute the reward of the generated caption using reward network
        rewards = GetRewards(features, captions, reward_network)
        
        # Compute the value of a random state in the generation process
        values = value_network(features, captions[:, :random.randint(1, MAX_SEQ_LEN)])
        
        # Compute the loss for the value and the reward
        loss = criterion(values, rewards)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(value_network.state_dict(), network_paths["value_network"])
            progress.set_description_str('Training Value Network: Best Loss %s' % best_loss)

        value_writer.add_scalar('Value Network-loss', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        value_network.valrnn.hidden_cell[0].detach_()
        value_network.valrnn.hidden_cell[1].detach_()
        reward_network.rewrnn.hidden_cell.detach_()
    
    return value_network


def train_policy_network(train_data, network_paths, plot_dir, batch_size=256, epochs=100000):

    policy_network = PolicyNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.0001)

    policy_writer = SummaryWriter(log_dir=os.path.join(plot_dir, 'runs'))

    best_loss = float("inf")
    print_green(f'[Training] Training Policy Network')
    progress = tqdm(range(epochs), desc='Training Policy Network: Best Loss %s' % best_loss)

    for epoch in progress:
        captions, features, _ = get_coco_batch(train_data, batch_size=batch_size, split='train')
        features = torch.tensor(features, device=device).float().unsqueeze(0)
        captions_in = torch.tensor(captions[:, :-1], device=device).long()
        captions_out = torch.tensor(captions[:, 1:], device=device).long()
        output = policy_network(features, captions_in)

        loss = 0
        for i in range(batch_size):
            # '2' is the end of segment, hence points to caption length
            caplen = np.nonzero(captions[i] == 2)[0][0] + 1
            loss += (caplen / batch_size) * criterion(output[i][:caplen], captions_out[i][:caplen])

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(policy_network.state_dict(), network_paths["policy_network"])
            progress.set_description_str('Training Policy Network: Best Loss %s' % best_loss)

        policy_writer.add_scalar('Policy Network-loss', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy_network


def train_reward_network(train_data, network_paths, plot_dir, batch_size=256, epochs=50000):

    reward_writer = SummaryWriter(log_dir = os.path.join(plot_dir, 'runs'))
    reward_network = RewardNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
    optimizer = optim.Adam(reward_network.parameters(), lr=0.0001)

    best_loss = float('inf')
    print_green(f'[Training] Training Reward Network')
    progress = tqdm(range(epochs), desc='Training Reward Network: Best Loss %s' % best_loss)

    for epoch in progress:

        captions, features, _ = get_coco_batch(train_data, batch_size=batch_size, split='train')
        features = torch.tensor(features, device=device).float()
        captions = torch.tensor(captions, device=device).long()
        ve, se = reward_network(features, captions)
        loss = VisualSemanticEmbeddingLoss(ve, se)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(reward_network.state_dict(), network_paths["reward_network"])
            progress.set_description_str('Training Reward Network: Best Loss %s' % best_loss)

        reward_writer.add_scalar('Reward Network-loss', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_network.rewrnn.hidden_cell.detach_()

    return reward_network


def train_a2c_network(train_data, save_paths, network_paths, plot_dir, epoch_count, episodes, retrain_all=False, curriculum=None):
    
    model_save_path = save_paths["model_path"]
    results_save_path = save_paths["results_path"]

    if retrain_all:
        print_green(f'[Training] Training all the networks')
        reward_network = train_reward_network(train_data, network_paths, plot_dir)
        policy_network = train_policy_network(train_data, network_paths, plot_dir)
        value_network = train_value_network(train_data, network_paths, plot_dir)
        print_green(f'[Training] All networks trained')

    else:
        try:
            reward_network = RewardNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
            reward_network.load_state_dict(torch.load(network_paths["reward_network"], map_location=device))
            print(f'[Training] loaded reward network')
        except FileNotFoundError:
            print(f'[Training] reward network not found')
            reward_network = train_reward_network(train_data, network_paths, plot_dir)
        try:
            policy_network = PolicyNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
            policy_network.load_state_dict(torch.load(network_paths["policy_network"], map_location=device))
            print(f'[Training] loaded policy network')
        except FileNotFoundError:
            print(f'[Training] policy network not found')
            policy_network = train_policy_network(train_data, network_paths, plot_dir)
        try:
            value_network = ValueNetwork(train_data["word_to_idx"], pretrained_embeddings=train_data["embeddings"]).to(device)
            value_network.load_state_dict(torch.load(network_paths["value_network"], map_location=device))
            print(f'[Training] loaded value network')
        except FileNotFoundError:
            print(f'[Training] value network not found')
            value_network = train_value_network(train_data, network_paths, plot_dir)

    for param in reward_network.parameters():
        param.require_grad = False

    a2c_network = AdvantageActorCriticNetwork(value_network, policy_network).to(device)
    a2c_network.train(True)
    optimizer = optim.Adam(a2c_network.parameters(), lr=0.0001)

    print(f'[Training] train_data len = {len(train_data["train_captions"])}')
    print(f'[Training] episodes = {episodes}')
    print(f'[Training] epoch_count = {epoch_count}')

    if curriculum is None:
        a2c_network = a2c_training(train_data, a2c_network, reward_network, optimizer, plot_dir, episodes, epoch_count)
    else:
        a2c_network = a2c_curriculum_training(train_data, a2c_network, reward_network, optimizer, plot_dir, episodes, epoch_count, curriculum)

    torch.save(a2c_network.state_dict(), model_save_path)
    with open(results_save_path, 'a') as f:
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')
        f.write(str(a2c_network))
        f.write('\n' + '-' * 10 + ' network ' + '-' * 10 + '\n')

    return a2c_network


def a2c_training(train_data, a2c_network, reward_network, optimizer, plot_dir, episodes, epoch_count):

    a2c_train_writer = SummaryWriter(log_dir=os.path.join(plot_dir,'runs'))

    print_green(f'[Training] Training Advantage Actor-Critic Network')
    best_loss = float('inf')
    progress = tqdm(range(epoch_count), desc='Training A2C Network: Advantage: %s, Best Loss %s' % (None, best_loss))
    
    for epoch in progress:
        episodic_avg_loss = 0

        captions, features, _ = get_coco_batch(train_data, batch_size=episodes, split='train')
        features = torch.tensor(features, device=device).float()
        captions = torch.tensor(captions, device=device).long()

        captions_in = captions
        features_in = features

        values, probs = a2c_network(features_in, captions_in)

        probs = F.softmax(probs, dim=2)
        dist = probs.cpu().detach().numpy()[:,0]

        actions = []
        for i in range(dist.shape[0]):
            actions.append(np.random.choice(probs.shape[-1], p=dist[i]))
        actions = torch.from_numpy(np.array(actions))

        gen_cap = actions.unsqueeze(-1).to(device)
        try:
            captions_in = torch.cat((captions_in, gen_cap), axis=1)
        except:
            captions_in = torch.cat((captions_in, gen_cap.long()), axis=1)

        log_probs = torch.log(probs[:,0,:].gather(1, actions.view(-1,1).to(device)))

        rewards = GetRewards(features_in, captions_in, reward_network)

        advantage = values - rewards
        actorLoss = (-log_probs * advantage).mean()
        criticLoss = 0.5 * advantage.pow(2).mean()

        loss = actorLoss + criticLoss
        episodic_avg_loss = loss.mean().item()

        optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        optimizer.step()

        if episodic_avg_loss < best_loss:
            best_loss = episodic_avg_loss
        progress.set_description_str('Training A2C Network: Advantage: %s, Best Loss %s' % (advantage.mean(), best_loss))

        # Summary Writer
        a2c_train_writer.add_scalar('A2C Network-episodic-loss', episodic_avg_loss, epoch)
        a2c_train_writer.add_scalar('A2C Network-episodic-mean-rewards', rewards.mean(), epoch)
        a2c_train_writer.add_scalar('A2C Network-episodic-mean-advantage', advantage.mean(), epoch)

        a2c_network.value_network.valrnn.hidden_cell[0].detach_()
        a2c_network.value_network.valrnn.hidden_cell[1].detach_()

    return a2c_network


def a2c_curriculum_training(train_data, a2c_network, reward_network, optimizer, plot_dir, episodes, epoch_count, curriculum):

    a2c_train_curriculum_writer = SummaryWriter(log_dir=os.path.join(plot_dir,'runs'))

    print_green(f'[Training] Training Advantage Actor-Critic Network')
    print_green(f'[Training] mode set to curriculum training using levels: {curriculum}')

    for level in curriculum:

        print_green(f'[Training] Training curriculum level: {level}')
        best_loss = float('inf')
        progress = tqdm(range(epoch_count), desc='Training A2C Curriculum Level: %s, Advantage: %s, Best Loss: %s' % (level, None, best_loss))
        
        for epoch in progress:
            episodic_avg_loss = 0

            captions, features, _ = get_coco_batch(train_data, batch_size=episodes, split='train')
            features = torch.tensor(features, device=device).float()
            captions = torch.tensor(captions, device=device).long()

            log_probs = []
            values = []
            rewards = []
            caplen = np.nonzero(captions == 2)[:,1].max() + 1

            if (caplen - level > 1):
                captions_in = captions[:, :caplen-level]
                features_in = features

                for step in range(level):
                    value, probs = a2c_network(features_in, captions_in)
                    probs = F.softmax(probs, dim=2)

                    dist = probs.cpu().detach().numpy()[:,0]
                    actions = []
                    for i in range(dist.shape[0]):
                        actions.append(np.random.choice(probs.shape[-1], p=dist[i]))
                    actions = torch.from_numpy(np.array(actions))

                    gen_cap = actions.unsqueeze(-1).to(device)
                    captions_in = torch.cat((captions_in, gen_cap), axis=1)
                    log_prob = torch.log(probs[:,0,:].gather(1, actions.view(-1,1).to(device)))

                    reward = GetRewards(features_in, captions_in, reward_network)

                    rewards.append(reward)
                    values.append(value)
                    log_probs.append(log_prob)

                values = torch.stack(values, axis=1).squeeze().to(device)
                rewards = torch.stack(rewards, axis=1).squeeze().to(device)
                log_probs = torch.stack(log_probs, axis=1).squeeze().to(device)

                advantage = values - rewards 
                actorLoss = (-log_probs * advantage).mean(axis=1)
                criticLoss = 0.5 * advantage.pow(2).mean(axis=1)

                loss = actorLoss + criticLoss
                episodic_avg_loss = loss.mean().item()

                if episodic_avg_loss < best_loss:
                    best_loss = episodic_avg_loss
                progress.set_description_str('Training A2C Curriculum Level: %s, Advantage: %s, Best Loss: %s' % (level, advantage.mean(), best_loss))

                optimizer.zero_grad()
                loss.mean().backward(retain_graph=True)
                optimizer.step()

                # Summary Writer
                writer_var_name = 'A2C Curriculum' + ' Level-' + str(level) + '-loss'
                a2c_train_curriculum_writer.add_scalar(writer_var_name, episodic_avg_loss, epoch)
                writer_var_name = 'A2C Curriculum' + ' Level-' + str(level) + '-mean-rewards'
                a2c_train_curriculum_writer.add_scalar(writer_var_name, rewards.mean(), epoch)
                writer_var_name = 'A2C Curriculum' + ' Level-' + str(level) + '-mean-advantage'
                a2c_train_curriculum_writer.add_scalar(writer_var_name, advantage.mean(), epoch)

            a2c_network.value_network.valrnn.hidden_cell[0].detach_()
            a2c_network.value_network.valrnn.hidden_cell[1].detach_()

    return a2c_network


def test_a2c_network(a2c_network, test_data, image_caption_data, data_size, validation_batch_size=128):
    
    with torch.no_grad():

        a2c_network.train(False)

        real_captions_filename = image_caption_data["real_captions_path"]
        generated_captions_filename = image_caption_data["generated_captions_path"]
        image_url_filename = image_caption_data["image_urls_path"]

        real_captions_file = open(real_captions_filename, "a")
        generated_captions_file = open(generated_captions_filename, "a")
        image_url_file = open(image_url_filename, "a")

        captions_real_all, features_real_all, urls_all = get_coco_batch(test_data, batch_size=data_size, split='val')
        val_captions_lens = len(captions_real_all)

        for i in tqdm(range(0, val_captions_lens, validation_batch_size), desc='Testing model'):
            features_real = features_real_all[i:i + validation_batch_size - 1]
            captions_real = captions_real_all[i:i + validation_batch_size - 1]
            urls = urls_all[i:i + validation_batch_size - 1]

            gen_cap = GenerateCaptionsWithActorCriticLookAhead(features_real, captions_real, a2c_network.policy_network, a2c_network.value_network, most_likely=True)
            gen_cap_str = decode_captions(gen_cap, idx_to_word=test_data["idx_to_word"])
            real_cap_str = decode_captions(captions_real, idx_to_word=test_data["idx_to_word"])

            real_captions_file.write("\n".join(real_cap_str))
            generated_captions_file.write("\n".join(gen_cap_str))
            image_url_file.write("\n".join(urls))

            real_captions_file.flush()
            generated_captions_file.flush()
            image_url_file.flush()

            a2c_network.value_network.valrnn.hidden_cell[0].detach_()
            a2c_network.value_network.valrnn.hidden_cell[1].detach_()

        real_captions_file.close()
        generated_captions_file.close()
        image_url_file.close()

