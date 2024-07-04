import argparse
from demo import ChatBot

# st = "5+9/(2+4*5/(9+1)-1)*2"
# print(calculate(st))


def args_parser():
    parser = argparse.ArgumentParser(description='LLM with tools')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--query', type=list,
                        default=['我想知道5+9/(2+4*5/(9+1)-1)*2等于？', '你能帮我算一下43+7*9/100+(42-10)/11的结果吗？',
                                 '帮我计算一下9乘4加上10再除2的结果'],
                        help='query list')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()

    user_name = "提问者"
    bot_name = "助手"
    content = "你是一名辅助提问者回答问题的助手"
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
    botx1 = ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.01, functions=functions)
    botx2 = ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.01)

    for i, line in enumerate(args.query):
        print(f'Q{i}: {line}')
        # exit()
        func_url = 'http://127.0.0.1:8641/expression'
        reply = botx1.chat_with_tools(line, func_url)
        print(f'chat with tools: {reply}')
        botx1.reset_messages([])

        reply = botx2.chat(line)
        print(f'chat without tools: {reply}')
        botx2.reset_messages([])
