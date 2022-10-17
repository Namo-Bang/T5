import pandas as pd
import re


def split(st):
    st = st.strip()[1:-1].split(':')
    st = [i.strip() for i in st]
    return st


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    FP_list = []
    FN_list = []
    TP_list = []
    for idx, item in enumerate(predict_golden):
        predicts = item[0]
        predicts = [[x[0].lower(), x[1].lower()] for x in predicts]
        pred_char = [i[0] for i in predicts]

        labels = item[1]
        labels = [[x[0].lower(), x[1].lower()] for x in labels]
        label_char = [i[0] for i in labels]

        used_check_labels = [0] * len(labels)
        used_check_preds = [0] * len(predicts)

        for id_, ele in enumerate(predicts):
            tp_ = False
            for index, i in enumerate(label_char):
                if i in ele[0] or ele[0] in i:
                    unmatched = 0
                    pred_tokens = ele[0].split(' ')
                    label_tokens = i.split(' ')

                    if len(pred_tokens) >= len(label_tokens):
                        for p in pred_tokens:
                            if p not in label_tokens:
                                unmatched += 1
                    else:
                        for l in label_tokens:
                            if l not in pred_tokens:
                                unmatched += 1

                    # 불일치 토큰 2개 이하
                    if unmatched < 2:
                        # TP 중복 가산 방지
                        if used_check_labels[index] == 0:
                            tp_ = True
                            TP += 1
                            TP_list.append(ele)
                            used_check_labels[index] = 1
                            used_check_preds[id_] = 1
                            break
            if tp_ == False:
                FP_list.append(ele)
        FP += (len(used_check_preds) - sum(used_check_preds))

        used_check_labels = [0] * len(labels)
        used_check_preds = [0] * len(predicts)
        for id_, ele in enumerate(labels):
            tp_ = False
            for index, i in enumerate(pred_char):
                if i in ele[0] or ele[0] in i:
                    unmatched = 0
                    label_tokens = ele[0].split(' ')
                    pred_tokens = i.split(' ')

                    if len(pred_tokens) >= len(label_tokens):
                        for p in pred_tokens:
                            if p not in label_tokens:
                                unmatched += 1
                    else:
                        for l in label_tokens:
                            if l not in pred_tokens:
                                unmatched += 1

                    if unmatched < 2:
                        if used_check_preds[index] == 0:
                            tp_ = True
                            used_check_preds[index] = 1
                            used_check_labels[id_] = 1
                            break
            if tp_ == False:
                FN_list.append(ele)
        FN += (len(used_check_labels) - sum(used_check_labels))

        if idx % 100 == 0:
            print('pred:', predicts)
            print('gold:', labels)
    with open('./FP_list.txt', 'w') as f:
        for ele in FP_list:
            f.write(':'.join(ele) + '\n')
    with open('./FN_list.txt', 'w') as f:
        for ele in FN_list:
            f.write(':'.join(ele) + '\n')
    with open('./TP_list.txt', 'w') as f:
        for ele in TP_list:
            f.write(':'.join(ele) + '\n')

    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1


def main():
    result = pd.read_csv('./outputs/predictions5.csv', index_col=0)
    pattern = '<[^<>]+:[^<>]+>'
    pred = result['Generated Text'].apply(
        lambda x: re.findall(pattern, x.replace('<extra_id_0>', '').replace('<pad>', ''))).values.tolist()
    label = result['Actual Text'].apply(lambda x: re.findall(pattern, x.replace('<pad>', ''))).values.tolist()

    pred2 = [list(map(split, i)) for i in pred]
    label2 = [list(map(split, i)) for i in label]

    predict_golden = zip(pred2, label2)
    p, r, f = calculateF1(predict_golden)

    print(f"precision:{p}\nrecall:{r}\nf1:{f}")
    with open("./example2/best_val_result.txt", "w") as fp:
        fp.write(f"precision:{p}\nrecall:{r}\nf1:{f}")


if __name__ == '__main__':
    main()