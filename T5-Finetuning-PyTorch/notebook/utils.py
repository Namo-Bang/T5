import json
import re
import random
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

import pandas as pd


class PromptMaker():
    def __init__(self, args, example=None):
        random.seed(42)
        self.prompt_type = ['raw', 'description', 'cins', 'cins2', 'unified', 'driven', 'example', 'cins2']
        self.args = args
        self.task = '문장에서 각 Entity를 찾아 표시하는 task를 NER이라고 한다.'
        self.entity_describe = 'Entity는 PS, LC, OG, TI, DT, QT 총 6종류가 있으며, 각각 '
        self.entity_definition = 'PS는 사람, LC는 위치, OG는 기관, TI는 시간, DT는 날짜, QT는 수량을 의미한다.'
        self.constraint = '단, PS의 경우 지시대명사에는 표시하지 않으나 가명처리된 이름에는 표시한다. 또한, QT는 어떤 물건의 양 뿐 아니라 단위가 있는 모든 어절에 표시한다.'
        self.prompt = '주어진 input에서 앞에 정의한 Entity를 찾아 표시하시오.'

        self.preset = {
            'raw': '',
            'description': lambda x: self.entity_definition + '\ninput: ' + x + (
                '\nNER: ' if self.args.ner_position == 'input' else '\n'),
            'cins': lambda
                x: self.task + ' ' + self.entity_describe + ' ' + self.entity_definition + ' ' + self.constraint + '\n' + self.prompt + '\ninput: ' \
                   + x + ('\nNER: ' if self.args.ner_position == 'input' else '\n'),
            'cins2': lambda
                x: self.task + ' ' + self.entity_describe + ' ' + self.entity_definition + ' ' + self.constraint + '\n' + '\ninput: ' \
                   + x + ('\nNER: ' if self.args.ner_position == 'input' else '\n'),
            'driven': lambda
                x: 'QT:수량에 해당하는 단어 DT:날짜에 해당하는 단어 PS:인명에 관련된 단어 LC:위치에 해당하는 단어 TI:시간 관련 단어 OG: 조직 관련 단어 [data] ' + x + \
                   ('\nNER: ' if self.args.ner_position == 'input' else '\n')
        }

        if example is not None:
            self.example = example
            self.prompt_type.append('example')
            self.preset['example'] = lambda \
                x: self.task + ' ' + self.entity_describe + ' ' + self.entity_definition + ' ' + self.constraint + '\n' \
                   + (('-' * 20) + '예시' + ('-' * 20)) + '\n' + '\n'.join(
                [a.strip() + ' NER: ' + b.strip() for a, b in self.example]) \
                   + '\n' + x + (' NER: ' if self.args.ner_position == 'input' else ' ')

    def add_prompt_type(self, prompt_type, preset):
        self.prompt_type.append(prompt_type)
        self.preset[prompt_type] = preset

    def get_prompt(self, prompt_type, data):
        assert prompt_type in self.prompt_type, 'please check prompt type or add prompt type'
        return [(self.preset[prompt_type](i['input'].strip())) for i in data]

    def get_df(self, prompt_type, data):
        prompt = self.get_prompt(prompt_type, data)
        ner = [('NER: ' if self.args.ner_position == 'target' else '') + d['target'] for d in data]
        target = [('NER: ' if self.args.ner_position == 'target' else '') + d['text'] for d in data]
        df = pd.DataFrame({'prompt': prompt, 'target': target, 'ner': ner})
        return df


def preprocessing(texts):
    pattern = '<[^<>]+:[^<>]+>'
    processed = []
    for text in tqdm(texts):
        try:
            find = re.findall(pattern, text)
            entity = [{i.split(':')[0][1:]: i.split(':')[1][:-1]} for i in find if ':' in i]
            entity_reverse = [{i.split(':')[1][:-1]: i.split(':')[0][1:]} for i in find if ':' in i]
            processed.append({
                'text': text,
                'value': find,
                'entity': entity,
                'entity_reverse': entity_reverse,
                'input': '',
                'target': '[SEP]'.join(find),
                'desc_ner': '개체명 인식 결과: [SEP]' + '[SEP]'.join(find)
            })
        except:
            print(text)
    for text in tqdm(processed):
        t = deepcopy(text['text'])
        for idx, i in enumerate(text['value']):
            t = t.replace(i, list(text['entity'][idx].keys())[0])
        text['input'] = t
    return processed


def get_counter(train_processed):
    dic = {}
    for data in train_processed:
        for ent in data['entity_reverse']:
            key, value = list(ent.items())[0]
            dic[key] = dic.get(key, []) + [value]

    counter = {}
    for i in dic.keys():
        counter[i] = pd.Series(Counter(dic[i])).to_frame()
        counter[i].columns = [i]

    return dic, counter


def get_one_seen_word(dic, counter):
    one_seen_word = []

    for i in dic.keys():
        # display(counter[i].sort_values(i,ascending=False))
        print(f'# of only one seen in {i}:', len(counter[i][counter[i][i] == 1]))
        one_seen_word += list(counter[i][counter[i][i] == 1].index)

    return one_seen_word


def get_one_seen_index(train_processed):
    dic, counter = get_counter(train_processed)
    one_seen_word = get_one_seen_word(dic, counter)
    one_seen_index = []
    for idx, i in enumerate(train_processed):
        k = []
        for j in i['entity']:
            k += list(j.keys())
        if set(k).intersection(set(one_seen_word)):
            one_seen_index.append(idx)
    return one_seen_index, one_seen_word, dic, counter


def train_test_split(data, train_portion=0.9, seed=None, shuffle=True, return_split=False):
    print('Get one seen word')
    one_seen_index, _, _, _ = get_one_seen_index(data)

    time_index = [idx for idx, i in enumerate(data) if 'TI' in i['text']]
    index = list(range(len(data)))

    not_one_seen_time = list(set(index).difference(set(one_seen_index + time_index)))

    intersect = sorted(list(set(one_seen_index).intersection(set(time_index))))
    time_index = sorted(list(set(time_index).difference(set(intersect))))
    one_seen_index = sorted(list(set(one_seen_index).difference(set(intersect))))

    import random
    if seed:
        random.seed(seed)

    if shuffle:
        print('Shuffling')
        random.shuffle(not_one_seen_time)
        random.shuffle(time_index)
        random.shuffle(one_seen_index)

    tr_num = int(len(not_one_seen_time) * train_portion)
    train_idx, dev_idx = not_one_seen_time[:tr_num], not_one_seen_time[tr_num:]

    one_seen_num = int(len(one_seen_index) * train_portion)
    one_seen_train_idx, one_seen_dev_idx = one_seen_index[:one_seen_num], one_seen_index[one_seen_num:]

    ti_num = int(len(time_index) * train_portion)
    ti_train_idx, ti_dev_idx = time_index[:ti_num], time_index[ti_num:]

    inter_num = int(len(intersect) * train_portion)
    inter_train_idx, inter_dev_idx = intersect[:inter_num], intersect[inter_num:]

    print('Split')
    if return_split:
        ret = {
            'general_train': [data[i] for i in train_idx],
            'general_dev': [data[i] for i in dev_idx],
            'one_seen_train': [data[i] for i in one_seen_train_idx],
            'one_seen_dev': [data[i] for i in one_seen_dev_idx],
            'ti_train': [data[i] for i in ti_train_idx],
            'ti_dev': [data[i] for i in ti_dev_idx],
            'intersect_train': [data[i] for i in inter_train_idx],
            'intersect_dev': [data[i] for i in inter_dev_idx],
        }
        train = []
        dev = []
        for k in ret.keys():
            if 'train' in k:
                train += ret[k]
            elif 'dev' in k:
                dev += ret[k]

        ret['train'] = train
        ret['dev'] = dev

        return ret

    else:
        train = train_idx + one_seen_train_idx + ti_train_idx + inter_train_idx
        dev = dev_idx + one_seen_dev_idx + ti_dev_idx + inter_dev_idx

        return [data[i] for i in train], [data[i] for i in dev]


def get_low_seen_values(data, leq_standard=1, rounds=2):
    assert leq_standard < rounds, 'under standard must be lower than rounds!'

    dic, counter = get_counter(data)
    low_seen_value = {}

    for i in dic.keys():
        for j, num in counter[i][counter[i][i] <= leq_standard].reset_index().values:
            low_seen_value[i] = low_seen_value.get(i, []) + (f'<{j}:{i}>\t' * (rounds - num)).split('\t')[:-1]

    return low_seen_value


def entity_mixing(data, leq_standard=1, rounds=2, seed=42):
    data = deepcopy(data)
    low_seen_values = get_low_seen_values(data, leq_standard, rounds)
    augmented_text = []

    random.seed(seed)
    random.shuffle(data)
    for k in list(low_seen_values.keys()):
        random.shuffle(low_seen_values[k])

    while True:
        target_data = random.choice(data)
        num_pop = {}

        for k in list(low_seen_values.keys()):
            num_pop[k] = target_data['text'].count(k)

        candidate = {}
        for k, v in num_pop.items():
            candidate[k] = []
            if len(low_seen_values[k]) < v:
                candidate[k] = low_seen_values[k]
                low_seen_values[k] = []
            else:
                if v:
                    candidate[k] = low_seen_values[k][-v:]
                    low_seen_values[k] = low_seen_values[k][:-v]

        if not any(candidate.values()):
            continue

        target_text = deepcopy(target_data['text'])
        for value in target_data['value']:
            entity = value.split(':')[-1][:-1]
            if candidate[entity]:
                mixing = candidate[entity].pop()
                target_text = target_text.replace(value, mixing)

        augmented_text.append(target_text)

        if any(low_seen_values.values()):
            continue

        else:
            print(low_seen_values)
            break

    return preprocessing(augmented_text)