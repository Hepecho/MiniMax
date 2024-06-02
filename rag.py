import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import MiniMaxEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from gte import pipeline_se
import os
import argparse
from demo import ChatBot


bot = ChatBot([])
os.environ["MINIMAX_GROUP_ID"] = bot.group_id
os.environ["MINIMAX_API_KEY"] = bot.api_key
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def args_parser():
    parser = argparse.ArgumentParser(description='RAG')
    parser.add_argument('--load', action="store_true", help="load vectorstore from local")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--faiss_path', type=str, default='output/faiss_index', help='vectorstore path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs={"parse_only": bs4_strainer},
    # )
    # docs = loader.load()
    # print(docs[0].metadata)
    args = args_parser()
    with open('data/the_little_prince_chinese_v2.txt', 'r', encoding='ansi') as f:
        content = f.read()
    doc = Document(page_content=content, metadata={'source': 'txt'})
    docs = [doc]
    # print(len(docs[0].page_content))
    # print(docs[0].page_content[:500])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # print(len(all_splits))
    # print(len(all_splits[0].page_content))
    # print(all_splits[10].metadata)
    # 90
    # 431
    # {'source': 'txt', 'start_index': 2700}

    if args.load:
        print('loading...')
        vectorstore = FAISS.load_local(args.faiss_path, MiniMaxEmbeddings(), allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(all_splits, MiniMaxEmbeddings())
        vectorstore.save_local(args.faiss_path)

    # query = "What makes the Little Prince's rose important"

    # for i in range(5):
    #     print(ref_docs[i].page_content)

    user_name = "提问者"
    bot_name = "助手"
    content = "你是一名辅助提问者回答问题的助手，你的职责是使用检索到的资料来回答问题。"
    botx = ChatBot([], user_name=user_name, bot_name=bot_name, content=content, temperature=0.1)

    query = '什么让小王子的玫瑰如此重要?'
    ref_docs = vectorstore.similarity_search(query, k=5)
    refs = []
    for i, d in enumerate(ref_docs):
        refs.append(f'{i+1}.\n' + d.page_content)
    material = '【参考资料】' + "\n".join(refs)
    line = f'【问题】：{query}\n{material}\n【回答】：'
    # print(line)
    # exit()
    reply = botx.chat(line)
    print(reply)
