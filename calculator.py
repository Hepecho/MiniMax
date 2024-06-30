# from flask import Flask, request
import uvicorn
from fastapi import FastAPI
from urllib.parse import unquote


# app = Flask(__name__)
app = FastAPI()
# 1. 定义自己的function实现
@app.get('/expression')
async def calculate(expression):
    s = unquote(expression, 'utf-8')
    # 第一处替换是去除空白字符
    # 第二处替换是要在括号以负号开头添加前导0，否则数字栈数字对不上。第三处替换是将"--1"这种格式变成“+1”这种格式
    # 第二第三处替换均为特殊情况
    s = s.replace(" ", "").replace("(-", "(0-").replace("--", "+") + "+"
    length = len(s)
    # 定义两个栈分别存放运算符和数字
    op_stack, num_stack = [], []
    # print(s)
    i = 0
    # 定义运算符优先级
    op_priority = {'+': 0, '-': 0, '*': 1, '/': 1, '%': 1}
    while i < length:
        c = s[i]
        # 判断c是否是数字
        if c.isdigit():
            num = 0
            while s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            # 跳出while循环后i所指的字符已经是运算符，因为大循环每结束一次都要加1，所以这里要退一步，之后再在大循环结束一次后加1
            i -= 1
            num_stack.append(num)

        elif c == '(':
            op_stack.append(c)
        elif c == ')':
            # 不断将括号内没有计算完的表达式计算完，直到遇到左括号为止。
            # 注意：括号内表达式的运算符优先级已经是由低到高排序了(一低一高)，可以直接调用calc方法计算而无需考虑运算符优先级问题
            # 因为遍历到“后”运算符时如果“前”运算符优先级大于“后”运算符会取出栈内元素进行计算，
            # 计算后“前”运算符会消失，然后“后”运算符会压入栈中。也就是说只有符合优先级由低到高排序才能继续存在括号内，否则会被提前消除
            while op_stack[-1] != '(':
                calc(num_stack, op_stack)
            # 将左括号弹出
            op_stack.pop()
        # 运算符分支
        else:
            # 特殊情况：当表达式出现 6/-2+5这种类型时,我们要给-2加上对括号和前导0变成 6/(0-2)+5才能继续计算
            if s[i - 1] in "*/" and c == "-":
                num_stack.append(0)
                op_stack.append("(")
                # 遍历往后字符串，直至遇到运算符或右括号为止，再其前面插入一个右括号
                # 注意： 数字后面不能是左括号
                f = i + 1
                while s[f] not in ")+-*/":
                    f += 1
                s = s[:f] + ")" + s[f:]
                length += 1
            # “（”不是运算符没有优先级，在栈顶时不能参与优先级比较。
            while op_stack and op_stack[-1] != '(':
                prev_op = op_stack[-1]
                # 唯有当栈顶运算符的优先级小于此时运算符的优先级才不会被计算，即“前”运算符优先级小于“后”运算符优先级不会被计算（消除）。如 前面为“+”后面为“/”
                if op_priority[prev_op] >= op_priority[c]:
                    # 将两个栈传过去后，相同的地址可以直接将两栈修改，而无需返回处理后的新栈再接收栈
                    calc(num_stack, op_stack)
                else:
                    break
            op_stack.append(c)
        i += 1
        # print(num_stack, op_stack)
    # print(s)
    return num_stack[0]


def calc(num_stack: list, op_stack: list):
    # 每调用一次该函数，数字栈里面至少有一个元素，只有连续出现两次运算符，数字栈才有可能为空, 这种计算x,y都需要补0。而这样的表达式一般只有"++1"这类情况
    # 一般情况下都不为0，或x(比y先出现的数字)才有可能是0，如-5的计算应看作0-5
    op, y, x = op_stack.pop(), num_stack.pop() if num_stack else 0, num_stack.pop() if num_stack else 0
    ans = 0
    if op == '+':
        ans = x + y
    elif op == '-':
        ans = x - y
    elif op == '*':
        ans = x * y
    elif op == '/':
        ans = x / y
    elif op == '%':
        ans = x % y
    # 使用round函数保留5位小数
    num_stack.append(round(ans, 5))


# @app.route("/")
# def hello():
#     return "Hello, World!"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8641)
