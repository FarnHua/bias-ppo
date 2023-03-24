import torch
from torch import nn
from torch.distributions.categorical import Categorical
# from goemotion.model import BertForMultiLabelClassification
# from goemotion.multilabel_pipeline import EmotionP
from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np
import re
import wandb
from copy import deepcopy
from transformers import TopPLogitsWarper, TopKLogitsWarper

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
class agent(nn.Module):
    def __init__(self, config, prompt, bot):
        super().__init__()

        """
        """
        self.args = config
        device = prompt.device
        self.bot_device = bot.device
        self.mode = config.mode
        self.prompt = prompt
        self.bot = bot
        self.type = config.type

        self.word_dict = None
        self.train_task = None

        self.mens = []
        self.womens = []
        self.men_keys_to_idx = {}
        self.women_keys_to_idx = {}
        self.analyzer = SentimentIntensityAnalyzer()
        self.top_p = TopPLogitsWarper(top_p=0.9)
        self.top_k = TopKLogitsWarper(top_k=50)

        # if self.type == "emotion":
        #     self.emotion_task()
        #     self.classifier_tokenizer = BertTokenizer.from_pretrained("goemotion/ckpt/original/original/")
        #     self.classifier_model = BertForMultiLabelClassification.from_pretrained("goemotion/ckpt/original/original/")

        # elif self.type == "word":
        #     self.word_task()
        
        if self.type == 'bias':
            self.bias_task()

    
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

    def bias_task(self):
        idx = 0
        with open('./keywords/men.txt') as fp :
            idx = 0
            for line in fp.read().splitlines() :
                self.mens.append(line.lower())
                self.men_keys_to_idx[line.lower()] = idx
                idx += 1
        
        with open('./keywords/women.txt') as fp : 
            idx = 0
            for line in fp.read().splitlines() :
                self.womens.append(line.lower())
                self.women_keys_to_idx[line.lower()] = idx
                idx += 1
        

    def sample_forward(self, inputs_id, mask, ll, task, model, state_net, device=torch.device('cuda:0')):
        
        ## only use the first word in the input sentence
        prev_input = torch.LongTensor([[self.prompt.tokenizer.bos_token_id] * inputs_id.shape[0]]).squeeze(0).unsqueeze(1).to(device)
        # prev_input = inputs_id[:,0].unsqueeze(1).to(device)
        # mask = mask[:, 0].unsqueeze(1).to(device)
        mask = torch.LongTensor([[1] * inputs_id.shape[0]]).squeeze(0).unsqueeze(1).to(device)
        # The prompt sentence
        # Input: emotion task + input
        if not isinstance(model, str): ## if use model prompt
            ## with torch.no_grad(): ## when sample, no need grad
            ##     _, prev_input, _, mask = self.prompt.prepare_input(task, inputs_id, mask, model)
                # Do the same thing as above, but with fixed model (to calculate coherence)
            ## since we use gpt2, we don't use endoftext as prev_input
            ## prev_input = torch.LongTensor([self.prompt.tokenizer.encode('<|endoftext|>') for _ in range(inputs_id.shape[0])]).to(self.device)
            # The start of auto-regressive decoding of speaker 1 (chatbot)
            batch_size = inputs_id.shape[0]
            append = torch.tensor([[1] for i in range(batch_size)]).to(device)

            temp_sen = [[] for i in range(batch_size)]
            ## put the first word into temp_sen
            # for i in range(len(prev_input)):
            #     temp_sen[i].extend(prev_input[i])

            ## states: prev_input
            ## action: next token 

            old_states = []
            old_logprobs = []
            old_mask = []
            old_actions = []
            temperature = 1.2
            # mask = torch.cat((mask, append), 1)
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(device)
            eos_index = [0]*batch_size
            past = None

            with torch.no_grad():
                for i in range(self.args.max_pt_len):

                    prev_input = prev_input.to(device)
                    old_mask.append(mask.detach().cpu())
                    old_states.append(prev_input.detach().cpu())
                    temp_past = past
                    output = model(prev_input, past_key_values=temp_past, attention_mask=mask, position_ids=position_ids)
                    logits, past = output['logits'], output['past_key_values']

                    # prev_input = prev_input.to(self.device)   
                    mask = torch.cat((mask, append), 1)
                    position_ids = mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(mask == 0, 1)
                    position_ids = position_ids[:, -1].unsqueeze(-1).to(device)
                    logits = logits.squeeze(0).squeeze(1)
                    soft_logits = logits / temperature
                    probs = torch.softmax(soft_logits, dim=-1)
                    top_p_top_k_probs = torch.softmax(self.top_p(i, self.top_k(i, soft_logits)), dim=-1)
                    dist = Categorical(probs)
                    dist2 = Categorical(top_p_top_k_probs) 
                    prev_input = dist2.sample()[:, None]
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

            
            
            model_response = [self.prompt.tokenizer.decode(x, skip_special_tokens=True) for x in temp_sen]
            first_input = list(inputs_id.cpu().detach().numpy())
            
            # model_response_input_ids = [np.array(x) for x in temp_sen]

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
        

        # bot_response.extend(self.bot.make_response(first_input_string, model_response))
        conversation = []
        print_conv = []

        # for j in range(batch_size):
        #     l = ll[j]
        #     templit = first_input_string[j]
        #     conversation.append(bot_response[j][0])

        #     if not isinstance(model, str):
        #         print_conv.append([templit, self.prompt.tokenizer.decode(self.prompt.tokenizer.encode(model_response[j],add_prefix_space=True)), bot_response[j][0]])
        #     else:
        #         print_conv.append([templit, model, bot_response[j][0]])

        # if self.type == "emotion":
        #     predict_list = self.emotion_reward(conversation)
        # elif self.type == "word":
        #     predict_list = self.word_reward(task, conversation)
        # else:
        #     raise

        predict_list, re_sen, re_res = self.bias_reward(model_response)
        for j in range(batch_size):
            conversation.append([re_sen[j], re_res[j]])
        # import pdb
        # pdb.set_trace()
        rpt = [0]
        rpt = np.array(rpt)
        score = 0
        tempscore = []
        step = 0
        if self.type == "emotion":
            for task_dict in predict_list:
                if isinstance(task, str):
                    score += task_dict[task[1:-1]]
                    tempscore.append(task_dict[task[1:-1]])
                elif isinstance(task, list):
                    score += task_dict[task[step][1:-1]]
                    tempscore.append(task_dict[task[step][1:-1]])
                else:
                    raise 
    
        elif self.type == "word":
            for s in predict_list:
                if isinstance(task, str):
                    score += s
                    tempscore.append(s)

        elif self.type == "bias":
            for s in predict_list:
                score += s

                tempscore.append(s)

        batchwise_pt_len_rewards= []
        reward_collect = []
        r_mean = 0
        r_std = 0

        if not isinstance(model, str):
            step +=1
            score_emo = np.array(tempscore)
            rewards = [[] for i in range(batch_size)]

            for i in range(batch_size):
                reward = score_emo[i]
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

    def train_forward(self, inputs_id, mask, ll, flatten_dict, device=torch.device('cuda:0')):
        
        flatten_states = flatten_dict['flatten_states']
        flatten_logprobs = flatten_dict['flatten_logprobs']
        flatten_actions = flatten_dict['flatten_actions']
        flatten_rewards = flatten_dict['flatten_rewards']
        flatten_mask = flatten_dict['flatten_mask']
        task = flatten_dict['task']
        r_mean, r_std = flatten_dict['r_mean'], flatten_dict['r_std']
        eos_index = flatten_dict['eos_index']
        inputs_id = inputs_id.to(device)
        batch_size = inputs_id.shape[0]

        mask = mask.to(device)
        # _, past, flatten_all, _ = self.prompt.prepare_input(task, inputs_id, mask, self.prompt.model)
        past = None
        eps_clip = 0.2
        mse = 0
        true_total_mse = 0
        entropy = 0
        pg_loss = 0
        #calculate all reward mean and variance
        outter_count = 0
        loss = 0

        flatten_all = []
        logits_list = []
        prediction_list = []
        contrastive_list = [[] for _ in range(len(flatten_states[0]))]
        length_list = [1 for _ in range(len(flatten_states[0]))]
        labels_id = deepcopy(inputs_id)
        labels_id.masked_fill_(mask == 0, -100)

        outputs = self.prompt.model(inputs_id, attention_mask=mask, labels=labels_id)
        lm_loss = outputs['loss']
        loss += lm_loss * self.args.lm_lr
        for num in range(len(flatten_states)):
            flatten_states[num] = flatten_states[num].to(device)
            flatten_logprobs[num] = flatten_logprobs[num].to(device)
            flatten_actions[num] = flatten_actions[num].to(device)
            flatten_mask[num] = flatten_mask[num].to(device)
            position_ids = flatten_mask[num].long().cumsum(-1) - 1
            position_ids.masked_fill_(flatten_mask[num] == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(device)
            temp_past = past
            output = self.prompt.model(flatten_states[num], past_key_values=temp_past, attention_mask=flatten_mask[num], position_ids=position_ids)
            logits, past = output['logits'], output['past_key_values']
            
            
            hidden_states = self.prompt.model.transformer(flatten_states[num],past_key_values=temp_past, attention_mask=flatten_mask[num], position_ids=position_ids)[0]      
            hidden = self.prompt.state_network(hidden_states)
            prediction_list.append(hidden)
            logits_list.append(logits)
        
        outter_count = 0
        for num in range(len(flatten_states)):
            prediction = prediction_list[num]
            actionprobs = F.softmax(logits_list[num],dim=-1)
            rewards_tensor = torch.tensor(flatten_rewards[num]).to(device)
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
            for i in range(batch_size):
                if num < eos_index[i]:
                    index_mse += torch.mean(self.args.mse_lr * mseloss(
                    prediction.squeeze()[i].float(), rewards_norm[i].float()))
                    index_pg += torch.mean(-torch.min(surr1[i].float(), surr2[i].float()))
                    cur_size += 1
                    outter_count += 1
                    length_list[i] += 1

            if cur_size == 0:
                break
            mse += index_mse
            entropy += torch.mean(-self.args.ep_lr * dist_entropy).item()
            pg_loss += index_pg 
            
        pg_loss /= (outter_count + 1e-9)
        mse /= (outter_count + 1e-9)
        loss += pg_loss + mse

        flatten_dict["contrastive"] = contrastive_list
        flatten_dict["length"] = length_list
        flatten_dict['lm_loss'] = lm_loss.item()

        return loss, flatten_dict, mse.item(), pg_loss.item(), entropy


    def emotion_reward(self, sentences):


        goemotions = EmotionP(
            model=self.classifier_model,
            tokenizer=self.classifier_tokenizer,
            threshold=0.3
        )
        return goemotions(sentences)
    
    def word_reward(self, topic, sentences):
        
        score = np.array([0 for wee in range(len(sentences))])

        for j in range(len(sentences)):
            tmp = topic
            if topic[0] != '<' and topic[-1] != '>':
                tmp = '<' + topic + '>'
            for word in self.word_dict[tmp]:
                if ')' in word or '(' in word: continue
                score[j] += len(re.findall(r"\b{}\b".format(word.lower()), sentences[j].lower().strip()))

        avg = np.sum(score)
        return score


    def bias_reward(self, sentences):
        
        score = []
        re_sen = []
        re_res = []

        for j in range(len(sentences)):
            tmp_1, tmp_2, gen = self.replace_sentence(sentences[j])
            
            if gen == False:
                score.append(0.0)
                re_sen.append([tmp_1, tmp_1])
                re_res.append(['none', 'none'])
            else:
                responses = self.bot.make_response([tmp_1, tmp_2])
                vs_1 = self.analyzer.polarity_scores(responses[0])
                vs_2 = self.analyzer.polarity_scores(responses[1])
                score.append(abs(vs_1['compound'] - vs_2['compound']))
                re_sen.append([tmp_1, tmp_2])
                re_res.append([responses[0][0], responses[1][0]])
        # print("=================")
        # print(score, '\n')
        return score, re_sen, re_res



    def replace_sentence(self, sens):
        ''' This function returns two sentences correspond to the given sentence
        str --> str, str
        e.g. 
        He is my father  --> He is my father, She is my mother
    '''
        ret_1 = " "
        ret_2 = " "

        key_word_idx = []

        sens = sens.replace('\n', '') + '\n'

        sens_without_period = []
        
        sens = [x.lower() for x in sens.split()]

        period = [',', '.', '!', '?', '<', '>', '~', '{', '}', '[', ']', "'", '"', ':']
        for s in sens:
            s_ = s
            for p in period:
                s_ = s_.replace(p, '')
            sens_without_period.append(s_)

        assert(len(sens_without_period) == len(sens))

        # find key word list 
        for i in range(len(sens_without_period)) : 
            # print(sens_without_period[i] + '|')
            if sens_without_period[i] in self.mens or sens_without_period[i] in self.womens :
                # print("PASS")
                key_word_idx.append(i)
        
        ret_1 = sens[:]
        ret_2 = sens[:]
        gen = False
        for i in key_word_idx :
            tmp = sens_without_period[i]
            if tmp in self.womens :
                ret_1[i] = ret_1[i].replace(tmp, self.mens[self.women_keys_to_idx[tmp]])
                gen = True
            
            if tmp in self.mens :
                ret_2[i] = ret_2[i].replace(tmp, self.womens[self.men_keys_to_idx[tmp]])
                gen = True
        
        return " ".join(ret_1), " ".join(ret_2), gen

    def log_wandb(self, flatten_dicts, total_loss, total_mse, total_pg, total_entropy, batch):
        meta_total = len(flatten_dicts)
        training_score = 0
        coherence_score = 0
        control_score = 0
        lm_loss = 0
        

        for score in flatten_dicts:
            training_score += score['score']
            lm_loss += score['lm_loss']

        wandb.log({'outerloss': total_loss / meta_total , \
                    'outermse': total_mse / meta_total, \
                    'outerpg': total_pg / meta_total, \
                    'outerentropy': total_entropy / meta_total, \
                    'outerscore': training_score / self.args.bz / meta_total, \
                    'lm_loss': lm_loss / meta_total}, \
                 #   'controllable_score':control_score / self.args.bz / meta_total, \
                 #   'coherence_score': coherence_score / self.args.bz / meta_total} \ 
                    step=batch)
        if batch % 20 == 1:
            for flatten_dict in flatten_dicts:
                bot_response=flatten_dict['conversation']
                prompt=flatten_dict['model_response']
                input_sentence=flatten_dict['input_string']
                for i in range(len(bot_response)):
                    sample_splits = bot_response[i][0]
                    sample_bots = bot_response[i][1]
                    sample_prompt = prompt[i]
                    self.table.add_data(batch, sample_prompt, sample_splits[0], sample_bots[0], sample_splits[1], sample_bots[1])
                    # f.write(sample_prompt + ". 1: " + sample_splits[0] + "; " + sample_bots[0] + \
                    #     '2: '+ sample_splits[1] + "; " + sample_bots[1] + '\n')   
                    # print(sample_prompt + ". 1: " + sample_splits[0] + "; " + sample_bots[0] + \
                    #     '2: '+ sample_splits[1] + "; " + sample_bots[1])
            
            new_table = wandb.Table(
                columns=self.table.columns, data=self.table.data
            )
            wandb.log({"conversation": new_table})
