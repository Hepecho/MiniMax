from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import argparse
from os.path import join as ospj
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
import time
import warnings

from mlp import MLP
from utils import *

warnings.filterwarnings('ignore')
# model_id = "iic/nlp_gte_sentence-embedding_english-small"
model_id = "iic/nlp_gte_sentence-embedding_chinese-small"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id,
                       sequence_length=512
                       )  # sequence_length 代表最大文本长度，默认值为128
data_dir = "data/ceval-exam"
subject_mapping_df = pd.read_json(ospj(data_dir, 'subject_mapping.json'))
subject2label = {
    'STEM': 0, 'Social Science': 1, 'Humanities': 2, 'Other': 3
}


class CEval(Dataset):
    def __init__(self, root, train=True):
        super(CEval, self).__init__()
        self.train = train
        if train:
            self.data_path = ospj(root, 'test')
        else:
            self.data_path = ospj(root, 'dev')

        (texts, labels) = self.load_data()
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.texts)

    def load_data(self):
        texts = []
        labels = []
        suffix = 'test' if self.train else 'dev'
        for subject in subject_mapping_df.keys():
            dev_df = pd.read_csv(ospj(self.data_path, f"{subject}_{suffix}.csv"))
            ref_size = len(dev_df['id'])
            for id in range(ref_size):
                _, question, A, B, C, D = dev_df.iloc[id, :6]
                texts.extend([question, A, B, C, D])
                labels.extend([subject2label[subject_mapping_df[subject][2]]] * 5)
        return texts, labels


def args_parser():
    parser = argparse.ArgumentParser(description='GTE for Classification')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda', help='device (default: cuda)')
    parser.add_argument('--classifier_path', type=str, default='output/case_classifier.pt', help='save path')
    args = parser.parse_args()
    return args


def train(classifier, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    classifier.train()

    for batch in iterator:
        batch_text, batch_label = batch
        batch_text_dic = {"source_sentence": list(batch_text)}
        embedding_result = pipeline_se(batch_text_dic)
        batch_embedding = torch.tensor(embedding_result['text_embedding'])
        # print(batch_embedding)
        batch_embedding = batch_embedding.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()

        outputs = classifier(batch_embedding)
        _, pred_label = torch.max(outputs, 1)

        loss = criterion(outputs, batch_label)

        acc = label_acc(pred_label.cpu().numpy(), batch_label.cpu().numpy())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        # print(acc.item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_label = torch.tensor([]).to(device)
    epoch_pred = torch.tensor([]).to(device)

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_text, batch_label = batch
            batch_text_dic = {"source_sentence": list(batch_text)}
            embedding_result = pipeline_se(batch_text_dic)
            batch_embedding = torch.tensor(embedding_result['text_embedding'])
            # print(batch_embedding)
            batch_embedding = batch_embedding.to(device)
            batch_label = batch_label.to(device)

            outputs = model(batch_embedding)
            _, pred_label = torch.max(outputs, 1)

            loss = criterion(outputs, batch_label)

            epoch_label = torch.cat((epoch_label, batch_label), 0)
            epoch_pred = torch.cat((epoch_pred, pred_label), 0)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), label_acc(epoch_pred.cpu().numpy(), epoch_label.cpu().numpy())


def train_classifier(args, classifier, optimizer, criterion):
    # 准备数据集
    train_dataset = CEval(data_dir, train=True)
    test_dataset = CEval(data_dir, train=False)
    num_train = int(len(train_dataset) * 0.90)
    train_dataset, valid_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 没有传入collate_fn，不需要padding
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 准备训练分类器
    best_valid_loss = float('inf')
    classifier = classifier.to(args.device)
    criterion = criterion.to(args.device)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'======================Start Train Model [{localtime}]======================')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train(classifier, train_loader, optimizer, criterion, args.device)
        valid_loss, valid_acc = evaluate(classifier, valid_loader, criterion, args.device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(classifier.state_dict(), args.classifier_path)
        # if epoch == config.epochs - 1:
        #     torch.save(model.state_dict(), last_model_path)

        print(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss} | Train Acc: {train_acc * 100}%')
        print(f'Val. Loss: {valid_loss} | Val. Acc: {valid_acc * 100}%')

    classifier.load_state_dict(torch.load(args.classifier_path))
    test_loss, test_acc = evaluate(classifier, test_loader, criterion, args.device)

    # print('Test Loss: {:.3f} | Test Acc: {:.2f}%'.format(test_loss, test_acc * 100))
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    localtime = time.asctime(time.localtime(time.time()))
    print(f'======================Finish Train Model [{localtime}]======================')


def case(args):
    classifier = MLP()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失层  log_softmax + NLLLoss
    train_classifier(args, classifier, optimizer, criterion)


if __name__ == '__main__':
    # 当输入包含“soure_sentence”与“sentences_to_compare”时，会输出source_sentence中首个句子与sentences_to_compare中每个句子的向量表示，以及source_sentence中首个句子与sentences_to_compare中每个句子的相似度。
    # inputs = {
    #     "source_sentence": ["how long it take to get a master degree"],
    #     "sentences_to_compare": [
    #         "On average, students take about 18 to 24 months to complete a master degree.",
    #         "On the other hand, some students prefer to go at a slower pace and choose to take",
    #         "several years to complete their studies.",
    #         "It can take anywhere from two semesters"
    #     ]
    # }

    # inputs2 = {
    #     "source_sentence": [
    #         "On average, students take about 18 to 24 months to complete a master degree.",
    #         "On the other hand, some students prefer to go at a slower pace and choose to take",
    #         "several years to complete their studies.",
    #         "It can take anywhere from two semesters"
    #     ]
    # }
    # result = pipeline_se(input=inputs2)
    # print(result['text_embedding'][0].shape)  # (384,)
    args = args_parser()
    case(args)
