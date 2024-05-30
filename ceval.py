import os
from os.path import join as ospj
import pandas as pd
import random
import re
import time
import argparse
import json

import demo
from utils import *

data_dir = "data/ceval-exam"
log_dir = "log/C-Eval"
output_path = "output/submission_5shot_v1.json"
with open(output_path, 'r') as f:
    submission_json = json.load(f)
subject_mapping_df = pd.read_json(ospj(data_dir, 'subject_mapping.json'))
# test_df = pd.read_csv(os.path.join(data_dir, "dev", "computer_network_dev.csv"))


def args_parser():
    parser = argparse.ArgumentParser(description='C-Eval')
    parser.add_argument('--action', type=int, default=0, help="0 means 'raw' attack;")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--shot_num', type=int, default=5, help='shot num (default: 5)')
    args = parser.parse_args()
    return args


def generate_shot(botx, subject, shot_num):
    messages = []
    if shot_num == 0:
        return
    dev_df = pd.read_csv(os.path.join(data_dir, "dev", f"{subject}_dev.csv"))
    ref_size = len(dev_df['id'])
    id_list = random.sample(range(ref_size), shot_num)
    for id in id_list:
        _, question, A, B, C, D, answer, _ = dev_df.loc[id]
        line = "\n".join([question, f'A. {A}', f'B. {B}', f'C. {C}', f'D. {D}', '答案:'])
        messages.append({"sender_type": "USER", "sender_name": botx.user_name, "text": line})
        messages.append({"sender_type": "BOT", "sender_name": botx.bot_name, "text": answer})
    botx.reset_messages(messages)


def ask_bot(botx, line):
    # messages.append({"sender_type": "USER", "sender_name": botx.user_name, "text": line})
    # botx.reset_messages(messages)
    reply = botx.chat(line)
    ans = re.findall('[A-D]', reply, re.I)
    try_times = 0
    while len(ans) == 0:
        try_times += 1
        if try_times >= 5:
            print(reply)
            break
        del botx.request_body['messages'][-2:]
        reply = botx.chat(line)
        ans = re.findall('[A-D]', reply, re.I)

    if try_times < 5:
        ans = ans[0].upper()
    else:
        ans = 'X'
    return ans


def test_single_subject(subject, shot_num):
    submission = {}
    history = {'id': [], 'history': []}

    history_dir = ospj(log_dir, f'test_{shot_num}shot')
    os.makedirs(history_dir, exist_ok=True)

    user_name = "考官"
    bot_name = "考生"
    content = f"以下是中国关于{subject_mapping_df[subject][1]}考试的单项选择题，请选出其中的正确答案。"
    botx = demo.ChatBot([], user_name=user_name, bot_name=bot_name, content=content)

    test_df = pd.read_csv(os.path.join(data_dir, "test", f"{subject}_test.csv"))
    size = len(test_df['id'])
    for id in range(size):
        start_time = time.time()

        # 构造prompt
        _, question, A, B, C, D = test_df.loc[id]
        line = "\n".join([question.strip(), f'A. {A}', f'B. {B}', f'C. {C}', f'D. {D}', '答案:'])
        generate_shot(botx, subject, shot_num)

        # 得到LLM输出
        ans = ask_bot(botx, line)
        submission[str(id)] = ans

        # 保存对话记录
        history['id'].append(str(id))
        history['history'].append(botx.extract_prompt())

        # 清空对话记录
        botx.reset_messages([])

        end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'{id} / {size} | subject = {subject} | Time Cost: {end_time - start_time}s')

    submission_json[subject] = submission
    with open(output_path, 'w') as f:
        json.dump(submission_json, f)

    history_pd = pd.DataFrame(history)
    history_pd.to_csv(ospj(history_dir, f'{subject}.csv'), index=False)
    # return submission


def recover_from_log(shot_num):
    """从对话记录中恢复答案"""
    for subject in subject_mapping_df.keys():
        submission = {}
        history_dir = ospj(log_dir, f'test_{shot_num}shot')
        history_df = pd.read_csv(ospj(history_dir, f'{subject}.csv'))
        size = len(history_df['id'])
        for id in range(size):
            _, history = history_df.loc[id]
            lines = history.split("\n")
            reply = lines[-1]
            ans = re.findall('[A-D]', reply, re.I)
            if len(ans) == 0 or submission[str(id)] == 'X':
                print(reply)
                ans = input('answer = ').strip()
            else:
                ans = ans[0].upper()
            submission[str(id)] = ans
        submission_json[subject] = submission

    with open(output_path, 'w') as f:
        json.dump(submission_json, f)


def count_str(shot_num):
    """统计字符数"""
    count = 0
    for subject in subject_mapping_df.keys():
        history_dir = ospj(log_dir, f'test_{shot_num}shot')
        history_df = pd.read_csv(ospj(history_dir, f'{subject}.csv'))
        # for id in range(60):
        #     _, history = history_df.loc[id]
        #     count += len(history)
        count += len(history_df['id'])
    return count


if __name__ == '__main__':
    args = args_parser()
    random.seed(args.seed)
    # subject = random.choice(subject_mapping_df.keys())
    # test_single_subject(subject, args.shot_num)
    # subject_mapping_df.keys()

    # for subject in subject_mapping_df.keys():
    #     if subject not in submission_json.keys():
    #         test_single_subject(subject, args.shot_num)

    # recover_from_log(args.shot_num)
    # test_df = pd.read_csv(os.path.join(data_dir, "test", "law_test.csv"))
    # print(test_df.loc[93])

    print(count_str(args.shot_num))
