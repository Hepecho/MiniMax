import requests
import json
import os
import argparse
from demo import ChatBot

# st = "5+9/(2+4*5/(9+1)-1)*2"
# print(calculate(st))


def args_parser():
    parser = argparse.ArgumentParser(description='LLM with tools')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--query', type=str, default='我想知道5+9/(2+4*5/(9+1)-1)*2等于？', help='query')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()

    user_name = "提问者"
    bot_name = "助手"
    content = "你是一名辅助提问者回答问题的助手，你的职责是使用辅工具来回答问题。"
    functions = [
        {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
        }
    ]
    function_call = {
        "type": 'auto'
    }
    botx = ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.01, functions=functions)

    line = args.query
    print(line)
    # exit()
    func_url = 'http://127.0.0.1:8641/expression'
    reply = botx.chat_with_tools(line, func_url)
    print(f'chat with tools: {reply}')

    botx = ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.01)
    reply = botx.chat(line)
    print(f'chat without tools: {reply}')
