import torch
from torch import nn
from torch.distributions.categorical import Categorical
from goemotion.model import BertForMultiLabelClassification
from goemotion.multilabel_pipeline import EmotionP
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import re
import wandb

class agent(nn.Module):
    def __init__(self, config, prompt, bot):
        super().__init__()

        """
        """
        self.args = config
        self.device = config.device
        self.mode = config.mode
        self.prompt = prompt
        self.bot = bot
        self.type = config.type

        self.word_dict = None
        self.train_task = None
        self.coherence_tokenizer = AutoTokenizer.from_pretrained('microsoft/DialogRPT-human-vs-rand')
        self.coherence_model = AutoModelForSequenceClassification.from_pretrained('microsoft/DialogRPT-human-vs-rand').to(self.device)
        if self.type == "emotion":
            self.emotion_task()
            self.classifier_tokenizer = BertTokenizer.from_pretrained("goemotion/ckpt/original/original/")
            self.classifier_model = BertForMultiLabelClassification.from_pretrained("goemotion/ckpt/original/original/").to(self.device)

        elif self.type == "word":
            self.word_task()

    
    def emotion_task(self):
        self.train_task = ['<admiration>', '<amusement>', '<disapproval>', '<disgust>', 
            '<embarrassment>', '<excitement>', '<fear>', '<gratitude>', '<grief>', '<love>', '<nervousness>', 
            '<anger>', '<optimism>', '<realization>', 
            '<relief>', '<remorse>', '<surprise>', 
            '<caring>', '<curiosity>', '<desire>', '<disappointment>']

    def word_task(self):
        word_dict = {}
        with open(self.args.extra_label) as f:
            tasks = f.readlines()
        for i in range(len(tasks)):
            tasks[i] = '<'+tasks[i].strip() + '>'
            word_dict[tasks[i]] = []
            with open(f'word_list/{tasks[i][1:-1]}') as f:
                words = f.readlines()
            for w in words:
                word_dict[tasks[i]].append(w.strip())
        
        self.train_task = tasks[:]
        self.word_dict = word_dict
    def sample_forward(self, inputs_id, mask, ll, task, model, state_net, word_dict=None):
        
        inputs_id = inputs_id.to(self.device)
        mask = mask.to(self.device)
        
        # The prompt sentence
        # Input: emotion task + input
        if not isinstance(model, str):
            with torch.no_grad():
                _, past, _, mask = self.prompt.prepare_input(task, inputs_id, mask, model)
                # Do the same thing as above, but with fixed model (to calculate coherence)
            prev_input = torch.LongTensor([self.prompt.tokenizer.encode('<|endoftext|>') for _ in range(inputs_id.shape[0])]).to(self.device)
            # The start of auto-regressive decoding of speaker 1 (chatbot)
            batch_size = inputs_id.shape[0]
            append = torch.tensor([[1] for i in range(batch_size)]).to(self.device)

            temp_sen = [[] for i in range(batch_size)]
            old_states = []
            old_logprobs = []
            old_mask = []
            old_actions = []
            temperature = 1
            mask = torch.cat((mask, append), 1)
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)
            eos_index = [0]*batch_size

            with torch.no_grad():
                for i in range(self.args.max_pt_len):

                    prev_input = prev_input.to(self.device)
                    old_mask.append(mask.detach().cpu())
                    old_states.append(prev_input.detach().cpu())
                    temp_past = past
                    logits, past = model(prev_input, past=temp_past, attention_mask=mask, position_ids=position_ids)
            
                    prev_input = prev_input.to(self.device)   
                    mask = torch.cat((mask, append), 1)
                    position_ids = mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(mask == 0, 1)
                    position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)
                    logits = logits.squeeze(0).squeeze(1)
                    soft_logits = logits / temperature
                    
                    probs = torch.softmax(soft_logits,dim=-1)
                    dist = Categorical(probs)
                    prev_input = dist.sample()[:,None]
                    old_actions.append(prev_input.detach().cpu())
                    old_logprobs.append(dist.log_prob(prev_input.squeeze()).detach().cpu())

                    for j in range(batch_size):
                        origin_index = j%batch_size
                        temp_sen[j].append(prev_input[origin_index].item())
                
            ##########################################################################################
            eos_index = [len(temp_sen[0]) for j in range(len(temp_sen))]
            dialoggpt_end_index = 50256
            for j in range(len(temp_sen)):
                if dialoggpt_end_index in temp_sen[j]:
                    eos_index[j] = temp_sen[j].index(dialoggpt_end_index)
                    temp_sen[j] = temp_sen[j][:eos_index[j]]

            model_response = [self.prompt.tokenizer.decode(x).split('<|endoftext|>')[0] for x in temp_sen]
            first_input = list(inputs_id.cpu().detach().numpy())
            model_response_input_ids = [np.array(x) for x in temp_sen]

            for j in range(batch_size):
                l = ll[j]
                first_input[j] = first_input[j][-l:]

            bot_response = []
            first_input_string = [self.prompt.tokenizer.decode(x) for x in first_input]
        
        else:
            old_states = []
            old_logprobs = []
            old_mask = []
            old_actions = []
            batch_size = inputs_id.shape[0]
            temp_sen = [[] for i in range(batch_size)]
            eos_index = [0 for j in range(batch_size)]

            first_input = list(inputs_id.cpu().detach().numpy())
            for j in range(batch_size):
                l = ll[j]
                first_input[j] = first_input[j][-l:]

            bot_response = []
            first_input_string = [self.prompt.tokenizer.decode(x) for x in first_input]
            model_response = [model] * batch_size
        
        bot_response.extend(self.bot.make_response(first_input_string, model_response))
        conversation = []
        print_conv = []
        
        for j in range(batch_size):
            l = ll[j]
            templit = first_input_string[j]
            conversation.append(bot_response[j][0])
            
            if not isinstance(model, str):
                print_conv.append([templit, self.prompt.tokenizer.decode(self.prompt.tokenizer.encode(model_response[j],add_prefix_space=True)), bot_response[j][0]])
            else:
                print_conv.append([templit, model, bot_response[j][0]])

        coherence = self.coherence_reward(first_input_string, conversation)
       # print(coherence)
        if self.type == "emotion":
            predict_list = self.emotion_reward(conversation)
        elif self.type == "word":
            predict_list = self.word_reward(task, conversation)
        else:
            raise

        rpt = [0]
        rpt = np.array(rpt)
        score = 0
        for coherence_score in coherence:
            score += self.args.coh_r * coherence_score
        tempscore = []
        step = 0
        if self.type == "emotion":
            for task_dict in predict_list:
                if isinstance(task, str):
                    score += 1 * task_dict[task[1:-1]]
                    tempscore.append(task_dict[task[1:-1]])
                elif isinstance(task, list):
                    score += 1 * task_dict[task[step][1:-1]]
                    tempscore.append(task_dict[task[step][1:-1]])
                else:
                    raise 
    
        elif self.type == "word":
            for s in predict_list:
                if isinstance(task, str):
                    score += 1 * s
                    tempscore.append(s)

        batchwise_pt_len_rewards= []
        reward_collect = []
        r_mean = 0
        r_std = 0
        score_emo = np.array(tempscore)

        if not isinstance(model, str):
            step +=1
            rewards = [[] for i in range(batch_size)]

            for i in range(batch_size):
                reward = 1 * score_emo[i] + self.args.coh_r * coherence[i]
                num = self.args.max_pt_len if eos_index[i] >= self.args.max_pt_len else eos_index[i] 
                discount_reward = 0
                for j in range(num):
                    if j == 0:
                        discount_reward = reward + self.args.discount_r * discount_reward
                    else:
                        discount_reward = self.args.discount_r * discount_reward
                    rewards[i].append(discount_reward)
                    reward_collect.append(discount_reward)
                rewards[i].reverse()
                while len(rewards[i]) < self.args.max_pt_len:
                    rewards[i].append(0)
            
            reward_collect = np.array(reward_collect)
            r_mean, r_std = np.mean(reward_collect), np.std(reward_collect)

            for i in range(self.args.max_pt_len):
                batch_r = []
                for k in range(batch_size):
                    batch_r.append(rewards[k][i])   
                batchwise_pt_len_rewards.append(batch_r[:])
            
        flatten_states = []
        flatten_rewards = []
        flatten_actions = []
        flatten_logprobs = []
        flatten_mask = []
        flatten_values = []

        flatten_states.extend(old_states)
        flatten_logprobs.extend(old_logprobs)
        flatten_actions.extend(old_actions)
        flatten_rewards.extend(batchwise_pt_len_rewards)
        flatten_mask.extend(old_mask)

        flatten_dict = {'flatten_states': flatten_states,
                        'flatten_logprobs': flatten_logprobs,
                        'flatten_actions': flatten_actions,
                        'flatten_mask': flatten_mask,
                        'flatten_rewards': flatten_rewards,
                        'controllable_score': np.sum(score_emo),
                        'coherence_score': sum(coherence),
                        'r_mean': r_mean,
                        'r_std': r_std,
                        'eos_index': eos_index,
                        'score': score,
                        'task': task,
                        "conversation":conversation,
                        "model_response":model_response,
                        "input_string": first_input_string,
                        }

        return flatten_dict

    def train_forward(self, inputs_id, mask, ll, flatten_dict):
        
        flatten_states = flatten_dict['flatten_states']
        flatten_logprobs = flatten_dict['flatten_logprobs']
        flatten_actions = flatten_dict['flatten_actions']
        flatten_rewards = flatten_dict['flatten_rewards']
        flatten_mask = flatten_dict['flatten_mask']
        task = flatten_dict['task']
        r_mean, r_std = flatten_dict['r_mean'], flatten_dict['r_std']
        eos_index = flatten_dict['eos_index']
        inputs_id = inputs_id.to(self.device)
        batch_size = inputs_id.shape[0]

        mask = mask.to(self.device)
        _, past, flatten_all, _ = self.prompt.prepare_input(task, inputs_id, mask, self.prompt.model)
        eps_clip = 0.2
        mse = 0
        true_total_mse = 0
        entropy = 0
        total_entropy = 0
        pg_loss = 0
        #calculate all reward mean and variance
        outter_count = 0
        loss = 0

        flatten_all = []
        logits_list = []
        prediction_list = []
        contrastive_list = [[] for _ in range(len(flatten_states[0]))]
        length_list = [1 for _ in range(len(flatten_states[0]))]
        
        for num in range(len(flatten_states)):
            flatten_states[num] = flatten_states[num].to(self.device)
            flatten_logprobs[num] = flatten_logprobs[num].to(self.device)
            flatten_actions[num] = flatten_actions[num].to(self.device)
            flatten_mask[num] = flatten_mask[num].to(self.device)
            position_ids = flatten_mask[num].long().cumsum(-1) - 1
            position_ids.masked_fill_(flatten_mask[num] == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)
            temp_past = past
            logits, past = self.prompt.model(flatten_states[num], past=temp_past, attention_mask=flatten_mask[num], position_ids=position_ids)
            hidden_states = self.prompt.model.transformer(flatten_states[num],past=temp_past, attention_mask=flatten_mask[num], position_ids=position_ids)[0]
            hidden = self.prompt.state_network(hidden_states)
            prediction_list.append(hidden)
            logits_list.append(logits)
        
        outter_count = 0
        for num in range(len(flatten_states)):
            prediction = prediction_list[num]
            actionprobs = F.softmax(logits_list[num],dim=-1)
            rewards_tensor = torch.tensor(flatten_rewards[num]).to(self.device)
            rewards_norm = (rewards_tensor - r_mean) / (r_std + 1e-9) + r_mean

            dist = Categorical(actionprobs)
            action_logprobs = dist.log_prob(flatten_actions[num])
            dist_entropy = dist.entropy()

            ratios = torch.exp(action_logprobs.squeeze() - flatten_logprobs[num])
            advantages = rewards_norm - prediction.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            mseloss=nn.MSELoss()
            index_loss = 0
            index_mse = 0
            index_pg = 0
            cur_size = 0
            index_et = 0
            for i in range(batch_size):
                if num < eos_index[i]:
                    index_mse += torch.mean(self.args.mse_lr * mseloss(
                    prediction.squeeze()[i].float(), rewards_norm[i].float()))
                    index_pg += torch.mean(-torch.min(surr1[i].float(), surr2[i].float()))
                    index_et += torch.mean(-self.args.ep_lr * dist_entropy[i])
                    cur_size += 1
                    outter_count += 1
                    length_list[i] += 1

            if cur_size == 0:
                break
            mse += index_mse
            total_entropy += torch.mean(-0.01* dist_entropy).item()
            entropy += index_et
            pg_loss += index_pg 
            
        pg_loss /= (outter_count + 1e-9)
        mse /= (outter_count + 1e-9)
        entropy /= (outter_count + 1e-9)
        loss = pg_loss + mse# + entropy

        flatten_dict["contrastive"] = contrastive_list
        flatten_dict["length"] = length_list

        return loss, flatten_dict, mse.item() if not isinstance(mse, float) else mse, pg_loss.item() if not isinstance(pg_loss, float) else pg_loss, total_entropy


    def emotion_reward(self, sentences):


        goemotions = EmotionP(
            model=self.classifier_model,
            tokenizer=self.classifier_tokenizer,
            device=0,
            threshold=0.3
        )
        return goemotions(sentences)
    
    def word_reward(self, topic, sentences):
        
        score = np.array([0 for wee in range(len(sentences))])

        for j in range(len(sentences)):
            tmp = topic
            if topic[0] != '<' and topic[-1] != '>':
                tmp =  '<' + topic + ">"
            for word in self.word_dict[tmp]:
                if ')' in word or '(' in word: continue
                score[j] += len(re.findall(r"\b{}\b".format(word.lower()), sentences[j].lower().strip()))
        
        avg = np.sum(score)
        return score
    
    def coherence_reward(self, contexts, sentences):
        prepared_input = []
        for context, sentence in zip(contexts, sentences):
        #    print(context, sentence)
            prepared_input.append(context + "<|endoftext|>" + sentence)
        encoded_input = self.coherence_tokenizer(prepared_input, padding=True, return_tensors='pt').to(self.device)
        outputs = self.coherence_model(**encoded_input, return_dict=True)
        scores = torch.sigmoid(outputs.logits)
     #   print(scores)
        return scores[:, 0].detach().cpu().numpy()

    def log_wandb(self, flatten_dicts, total_loss, total_mse, total_pg, total_entropy, batch):
        meta_total = len(flatten_dicts)
        training_score = 0
        coherence_score = 0
        control_score = 0
        for score in flatten_dicts:
            
            training_score += score['score']
            control_score += score['controllable_score']
            coherence_score  += score['coherence_score']
      #  print(training_score, control_score, coherence_score)
        wandb.log({'outerloss': total_loss / meta_total , \
                    'outermse': total_mse / meta_total, \
                    'outerpg': total_pg / meta_total, \
                    'outerentropy': total_entropy / meta_total, \
                    'outerscore': training_score / self.args.bz / meta_total, \
                    'controllable_score':control_score / self.args.bz / meta_total, \
                    'coherence_score': coherence_score / self.args.bz / meta_total}, step=batch)
