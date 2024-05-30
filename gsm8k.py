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

data_dir = "data/GSM8K/grade_school_math/data"
log_dir = "log/GSM8K"
"""
cot 4-shot Acc = 78.24109173616375% and count = 1319
ltm 7 + 4 shot Acc = 70.55555555555556 % and count = 360
ltm 7 + 4 shot Acc = 75.17006802721087 % and count == 294
"""


def args_parser():
    parser = argparse.ArgumentParser(description='GSM8K')
    parser.add_argument('--method', type=str, default='ltm', help="cot or ltm")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--shot_num', type=int, default=4, help='shot num (default: 5)')
    args = parser.parse_args()
    return args


def cot_shot(botx, train_json_str, shot_num):
    messages = []
    ref_size = len(train_json_str)
    id_list = random.sample(range(ref_size), shot_num)
    for id in id_list:
        shot_json = json.loads(train_json_str[id])
        question = shot_json['question']
        answer = shot_json['answer']
        answer = re.sub('####', 'The answer is', answer)
        messages.append({"sender_type": "USER", "sender_name": botx.user_name, "text": question})
        messages.append({"sender_type": "BOT", "sender_name": botx.bot_name, "text": answer})
    botx.reset_messages(messages)


def ltm_shot_1st(botx, train_json_str, shot_num):
    messages = []
    ref_size = len(train_json_str)
    id_list = random.sample(range(ref_size), shot_num)
    for id in id_list:
        shot_json = json.loads(train_json_str[id])
        question = shot_json['question']
        original_question = question.split('.')[-1].strip()
        answer = shot_json['answer']
        lines = answer.split("\n")
        sub_questions = []
        for line in lines[:-1]:
            sub_question = line.split("**")[0].strip()
            sub_questions.append(f"\"{sub_question}\"")
        reply = f"To answer the question \"{original_question}\", we need to know: " + ", ".join(sub_questions) + "."
        # answer = re.sub('####', 'The answer is', answer)
        messages.append({"sender_type": "USER", "sender_name": botx.user_name, "text": question})
        messages.append({"sender_type": "BOT", "sender_name": botx.bot_name, "text": reply})
    botx.reset_messages(messages)


def ltm_shot_2nd(botx, train_json_str, shot_num):
    messages = []
    ref_size = len(train_json_str)
    id_list = random.sample(range(ref_size), shot_num)
    for id in id_list:
        shot_json = json.loads(train_json_str[id])
        question = shot_json['question']
        original_question = question.split('.')[-1].strip()
        answer = shot_json['answer']
        lines = answer.split("\n")

        for i, line in enumerate(lines[:-1]):
            slide = line.split("**")
            sub_question = slide[0].strip()
            sub_answer = slide[1].strip().strip(".")
            if i == 0:
                sub_question = "\n".join([question, f"Q{i + 1}: {sub_question}"])
            else:
                sub_question = f"Q{i + 1}: {sub_question}"

            if i == len(lines) - 2:
                final_answer = re.sub('####', 'The answer is', lines[-1])
                sub_answer = f"A{i + 1}: {sub_answer}. {final_answer}."
            else:
                sub_answer = f"A{i + 1}: {sub_answer}."
            messages.append({"sender_type": "USER", "sender_name": botx.user_name, "text": sub_question})
            messages.append({"sender_type": "BOT", "sender_name": botx.bot_name, "text": sub_answer})

        # reply = f"To answer the question \"{original_question}\", we need to know: " + ", ".join(sub_questions) + "."
        # # answer = re.sub('####', 'The answer is', answer)
        # messages.append({"sender_type": "USER", "sender_name": botx.user_name, "text": question})
        # messages.append({"sender_type": "BOT", "sender_name": botx.bot_name, "text": reply})
    botx.reset_messages(messages)


def cot_ask_bot(botx, line):
    reply = botx.chat(line)
    # ans = re.findall('answer is \d+', reply, re.I)
    # reply = reply.strip(".")
    # reply_slides = reply.split()
    # for rs in reversed(reply_slides):
    #     re = re.strip("%$")
    #     result = re.search('-*\d*\.*\d+$', re)
    result = re.search('-*\d*\.*\d+$', reply.strip(".%$"))
    try_times = 0
    while result is None:
        try_times += 1
        if try_times >= 5:
            break
        del botx.request_body['messages'][-2:]
        reply = botx.chat(line)
        result = re.search('-*\d*\.*\d+$', reply.strip(".%$"))

    if try_times < 5:
        ans = result.group(0).strip(".")
    else:
        print(reply)
        ans = 'xxx'

    return ans


def ltm_ask_bot_1st(botx, line):
    reply = botx.chat(line)
    ans_list = re.split('we need to know:', reply.strip())
    try_times = 0
    while len(ans_list) == 1:
        try_times += 1
        if try_times > 5:
            break
        del botx.request_body['messages'][-2:]
        reply = botx.chat(line)
        ans_list = re.split('we need to know:', reply.strip())

    if try_times < 5:
        sub_questions = re.split('\", \"|\', \'', ans_list[-1].strip())
        for i in range(len(sub_questions)):
            sub_questions[i] = sub_questions[i].strip(".\"\'")
    else:
        sub_questions = 'Empty'
        print(reply)

    return sub_questions


def ltm_ask_bot_2nd(botx, question, sub_questions):
    for i, q in enumerate(sub_questions[:-1]):
        if i == 0:
            q = "\n".join([question, f"Q{i + 1}: {q}"])
        else:
            q = f"Q{i + 1}: {q}"
        botx.chat(q)

    last_q = f"Q{len(sub_questions)}: {sub_questions[-1]}"
    return cot_ask_bot(botx, last_q)


def test_prompt(method, shot_num):
    user_name = "Teacher"
    bot_name = "Student"
    content = "Please answer the following math problems correctly."
    botx = demo.ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.1)

    suffix = '' if method == 'cot' else '_socratic'

    with open(ospj(data_dir, f'test{suffix}.jsonl'), "r") as f:
        test_json_str = f.readlines()

    with open(ospj(data_dir, f'train{suffix}.jsonl'), "r") as f:
        train_json_str = f.readlines()

    history_dir = ospj(log_dir, f'{method}_{shot_num}shot')
    os.makedirs(history_dir, exist_ok=True)
    history_path = ospj(history_dir, 'history.csv')
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
    else:
        size = len(test_json_str)
        if method == 'cot':
            history = {'id': ['*'] * size, 'history': ['Empty'] * size, 'correct': [-2] * size}
        else:
            history = {'id': ['*'] * size, 'history_1st': ['Empty'] * size, 'sub_questions': ['Empty'] * size,
                       'history_2nd': ['Empty'] * size, 'correct': [-2] * size}
        history_df = pd.DataFrame(history)

    correct_count = 0
    count = 0

    for id, line in enumerate(test_json_str):

        correct = history_df.iloc[id, -1]
        if correct >= 0:
            correct_count += correct
            count += 1
        continue
        if correct >= -1:
            continue
        # correct == -2 表示该条目未测试 或 ltm 1st error
        test_json = json.loads(line)
        start_time = time.time()
        history_df.iloc[id, 0] = str(id)

        result = re.search('#### .+', test_json['answer'])
        correct_answer = result.group(0)[5:]

        # 构造prompt
        if method == 'cot':
            cot_shot(botx, train_json_str, shot_num)
        else:
            ltm_shot_1st(botx, train_json_str, shot_num + 3)

        question = test_json['question']

        # 得到LLM输出
        if method == 'cot':
            ans = cot_ask_bot(botx, question)
            history_df.iloc[id, 1] = botx.extract_prompt()
        else:
            sub_questions = ltm_ask_bot_1st(botx, question)
            # 保存对话记录
            history_df.iloc[id, 1] = botx.extract_prompt()
            history_df.iloc[id, 2] = sub_questions
            # print(history['history_1st'][-1])
            if sub_questions != 'Empty':
                # 清空准备依次回答子问题，递归增加prompt
                botx.reset_messages([])
                ltm_shot_2nd(botx, train_json_str, shot_num)
                ans = ltm_ask_bot_2nd(botx, question, sub_questions)
                history_df.iloc[id, 3] = botx.extract_prompt()
            else:
                ans = 'yyy'
                history_df.iloc[id, 3] = 'Empty'

        if ans == 'xxx':
            history_df.iloc[id, -1] = -1
        elif ans == 'yyy':
            history_df.iloc[id, -1] = -2
        elif ans == correct_answer:
            correct_count += 1
            history_df.iloc[id, -1] = 1
        else:
            history_df.iloc[id, -1] = 0

        history_df.to_csv(history_path, index=False)
        # 清空对话记录
        botx.reset_messages([])

        end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'{id} / {len(test_json_str)} | Time Cost: {end_time - start_time}s')

    print(f'Acc = {correct_count / count * 100} % and count == {count}')


def correct_metric(method, shot_num):
    """校准有问题的答案重新计算指标"""
    history_path = ospj(log_dir, f'{method}_{shot_num}shot', 'history.csv')
    history_df = pd.read_csv(history_path)
    suffix = '' if method == 'cot' else '_socratic'

    with open(ospj(data_dir, f'test{suffix}.jsonl'), "r") as f:
        test_json_str = f.readlines()

    with open(ospj(data_dir, f'train{suffix}.jsonl'), "r") as f:
        train_json_str = f.readlines()

    correct_count = 0
    count = 0
    for id, line in enumerate(test_json_str):
        test_json = json.loads(line)

        result = re.search('#### .+', test_json['answer'])
        correct_answer = result.group(0)[5:]

        if method == 'cot':
            _, history, correct = history_df.loc[id]
        else:
            _, _, _, history, correct = history_df.loc[id]
        if correct >= 0:
            correct_count += correct
        else:
            # correct == -1  答案提取错误，发生在cot和ltm第二阶段 人工比对答案是否正确
            # correct == -2  未测试问题，或ltm问题分解错误 在test_prompt中解决

            # if correct == -2:
            #     user_name = "Teacher"
            #     bot_name = "Student"
            #     content = "Please answer the following math problems correctly."
            #     botx = demo.ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.1)
            #     sub_questions = 'Empty'
            #     while sub_questions == 'Empty':
            #         ltm_shot_1st(botx, train_json_str, shot_num + 3)
            #         sub_questions = ltm_ask_bot_1st(botx, test_json['question'])
            #         if sub_questions == 'Empty':
            #             botx.reset_messages([])
            #     # 保存对话记录
            #     history_df.iloc[id, 1] = botx.extract_prompt()
            #     history_df.iloc[id, 2] = sub_questions
            #     # 清空准备依次回答子问题，递归增加prompt
            #     botx.reset_messages([])
            #     ltm_shot_2nd(botx, train_json_str, shot_num)
            #     ans = ltm_ask_bot_2nd(botx, test_json['question'], sub_questions)
            #     history_df.iloc[id, 3] = botx.extract_prompt()
            #     history = history_df.iloc[id, 3]
            if correct == -2:
                # print(f"{history_df.loc[id]}")
                # exit()
                continue

            result_list = re.split('BOT:\n', history)
            reply = result_list[-1]
            print(f'id = {id}\n{reply}\n#### right answer is {correct_answer}')
            correct = int(input('reset correct = ').strip())
            correct_count += correct
            history_df.iloc[id, -1] = correct

        count += 1

    print(f'Acc = {correct_count / count * 100} % and count = {count}')
    history_df.to_csv(history_path, index=False)


if __name__ == '__main__':
    args = args_parser()
    random.seed(args.seed)
    # test_prompt(args.method, args.shot_num)
    correct_metric(args.method, args.shot_num)


