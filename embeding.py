import torch
import matplotlib.pyplot as plt
from model import *
from data_process import *
import tqdm

def train(epoches,lr,model):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    tran_acc = []
    num_correct = 0
    num_trains = 0
    for epoch in tqdm.tqdm(range(epoches)):
        epoch_loss = 0
        for iter, (batched_graph, labels) in enumerate(train_dataloader):
            feats = batched_graph.ndata['sensitive']
            # torch.stack((batched_graph.ndata['sensitive'], batched_graph.ndata['entrypoint'], batched_graph.ndata['external']))
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            num_trains += len(labels)
            num_correct += (logits.argmax(1) == labels).sum().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}, train acc {:.3f}'.format(epoch, epoch_loss, num_correct / num_train))
        epoch_losses.append(epoch_loss)
        tran_acc.append(num_correct / num_train)
        num_correct = 0
        num_trains = 0
    test_acc = evaluate(model)
    with open('./result/para.txt','a') as file:
        file.write('epochs: {}, test acc: {}, lr: {}\n'.format(epoches,test_acc,lr))
    plt.plot(np.array([i for i in range(epoches)]), epoch_losses, color='g', label='loss')
    plt.plot(np.array([i for i in range(epoches)]), tran_acc, color='r', label='accuracy')
    plt.legend(loc=10, bbox_to_anchor=(0.65, 0.95), ncol=2, shadow=True, fancybox=True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss & Accuracy')
    plt.show()

def evaluate(model):
    model.eval()
    test_correct = 0
    test_nums = 0
    type_total = [0]*4
    type_correct = [0]*4
    type_pre_tol = [0]*4
    for test_graph, test_labels in test_dataloader:
        pred = model(test_graph, test_graph.ndata['sensitive'])
        predt = pred.argmax(1)
        for i,item in enumerate(test_labels):
            type_total[item] += 1
            type_pre_tol[predt[i]] += 1
            if predt[i] == item:
                type_correct[item] += 1
        test_correct += (pred.argmax(1) == test_labels).sum().item()
        test_nums += len(test_labels)
    print('Test acc: {:.4f}'.format(test_correct / test_nums))
    print(type_correct)
    print(type_total)
    print(type_pre_tol)
    return test_correct / test_nums

    # if test_correct / test_nums >= 0.75:
    #     torch.save(model, './result/GCN.pt')
    # with open('./result/para.txt','a') as file:
    #     file.write('epochs: {}, test acc: {}, lr: {}\n'.format(epoches,test_correct/test_nums,lr))


if __name__ == '__main__':
    model = Classifier(89, 89, len(gtype))
    #model = torch.load('./result/GCN.pt')
    train(160, 0.001, model)

