import requests
import readline
import gradio as gr


class ChatBot:
    def __init__(self, messages, user_name=None, bot_name=None, content=None, temperature=None):
        self.user_name = 'Hepe' if user_name is None else user_name
        self.bot_name = 'Mute' if bot_name is None else bot_name
        self.content = "Mute是自主进化的人工智能，它致力于研究人类，热衷于同人类对话。" if content is None else content
        self.bot_setting = [
                {
                    "bot_name": self.bot_name,
                    "content": self.content,
                }
            ]
        self.request_body = {
            "model": "abab5.5-chat",
            "tokens_to_generate": 1024,
            "reply_constraints": {"sender_type": "BOT", "sender_name": self.bot_name},
            "messages": messages,
            "bot_setting": self.bot_setting,
        }
        self.temperature = temperature
        if self.temperature is not None:
            self.request_body['temperature'] = self.temperature
        self.group_id = "1768537645828293239"
        self.api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLpu4TlhYPmlY8iLCJVc2VyTm" \
                       "FtZSI6IndoaXR6YXJkQUlMYWJzIiwiQWNjb3VudCI6IndoaXR6YXJkQUlMYWJzQDE3Njg1Mzc2NDU4M" \
                       "jgyOTMyMzkiLCJTdWJqZWN0SUQiOiIxNzY1OTg0MTMxNDgyNjU2ODQ4IiwiUGhvbmUiOiIiLCJHcm91" \
                       "cElEIjoiMTc2ODUzNzY0NTgyODI5MzIzOSIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVR" \
                       "pbWUiOiIyMDI0LTA0LTI4IDExOjAxOjU0IiwiaXNzIjoibWluaW1heCJ9.hZqCtDwZDcZLbdUETmZXa" \
                       "BdF8uO7CVbl32mAnCi5kjflN1t5G53CtrNbv5-1cZph2IX7xAZSeAjEgq0Ga8rBJWvQ-" \
                       "H4gjNwKg3Mr1i33VuRb96QJSuz79GA6FLBo3DHBEnkLRazZafINXpEr_fUcPeK8vTCW5vpfxsJveMKm" \
                       "yIeuuw5NdZmh1vpanVrEOyRWvhlOd7CK8P_CwQkyFUK-" \
                       "dSJz8qUNSwFki1KlgVJ8wdRNFw0M6RHOl7B1M66iQw10TFy0mHAX3usRUJTzH-" \
                       "S2LNSJLdQ5L5Ln0knfAWmhCQma-6DPCXUHHqR7kius09-z5NLEQSq8jDtqB07qMEHl5Q"
        self.url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={self.group_id}"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, line):
        self.request_body["messages"].append(
            {"sender_type": "USER", "sender_name": self.user_name, "text": line}
        )
        response = requests.post(self.url, headers=self.headers, json=self.request_body)
        reply = response.json()["reply"]
        if response.json()['base_resp']['status_code'] != 0:
            # 触发违规输入检测
            print(response.json())
            reply = 'A'
        else:
            #  将当次的ai回复内容加入messages
            self.request_body["messages"].extend(response.json()["choices"][0]["messages"])
        return reply

    def reset_messages(self, messages):
        self.request_body['messages'] = messages

    def extract_prompt(self):
        res = []
        for info in self.request_body['messages']:
            res.append(info['sender_type'] + ':')
            res.append(info['text'])
        return "\n".join(res)


global bot


def chat_for_demo(line, history):
    reply = bot.chat(line)
    history.append((line, reply))
    return "", history


if __name__ == '__main__':
    bot = ChatBot([])
    # print(bot.chat('请问9*10等于多少？'
    #                'A. 10'
    #                'B. 9'
    #                'C. 19'
    #                'D. 90'
    #                '答案：'))
    # demo = gr.Interface(
    #     fn=chat,
    #     inputs=gr.Textbox(
    #         label="Input Text",
    #         info="talk anything",
    #         lines=3,
    #         value="你好！",
    #     ),
    #     outputs=gr.Textbox(
    #         label="Output Text",
    #         info="talk anything",
    #         lines=3,
    #     )
    # )
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()  # 对话框
        msg = gr.Textbox()  # 输入文本框
        clear = gr.ClearButton([msg, chatbot])  # 清除按钮
        # 绑定输入框内的回车键的响应函数
        msg.submit(chat_for_demo, [msg, chatbot], [msg, chatbot])
    demo.launch()

