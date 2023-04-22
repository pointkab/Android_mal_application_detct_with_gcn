from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.data
from dgl.data import DGLDataset
import torch
import os
from sklearn.utils import shuffle
import numpy as np
from CGmain import *


save_dir = './data'
dir = ['D:\Compressed\Benign','D:\Compressed\Adware','D:\Compressed\Banking','D:\Compressed\Riskware']

gtype = {
    'Benign':0,
    'Adware':1,
    'Banking':2,
    'Riskware':3,
}

#自定义数据集
class MACDataset(DGLDataset):
    def __init__(self,
        url=None,
        save_dir=save_dir,
        raw_dir=save_dir,
        force_reload=False,
        verbose=False):
        super(MACDataset,self).__init__(name='macdata',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
    def download(self):
        pass
    def process(self):
        self.graphs = []
        self.labels = []
        gpath = os.listdir(self.raw_dir)
        for f in gpath:
            _graphs,_glabels = dgl.load_graphs(os.path.join(save_dir, f))
            self.graphs.extend(_graphs)
            self.labels.extend(_glabels['glabel'])
        self.graphs,self.labels = shuffle(self.graphs,self.labels,random_state=10)

    def __getitem__(self, idx):
        return self.graphs[idx],self.labels[idx]
    def __len__(self):
        return len(self.graphs)
    def save(self):
        pass
    def load(self):
        pass
    def has_cache(self):
        pass

def collate(samples):
    graphs,labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph,batched_labels

dataset =MACDataset()

num_examples = len(dataset)
num_train = int(num_examples*0.8)
train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train,num_examples))

#构建训练和测试数据
train_dataloader = GraphDataLoader(
    dataset,
    sampler=train_sampler,
    batch_size=16,
    collate_fn=collate,
    drop_last=False,
)

test_dataloader = GraphDataLoader(
    dataset,
    sampler=test_sampler,
    batch_size=16,
    collate_fn=collate,
    drop_last=False,
)

#获取文件的访问地址
def getapkpath(type):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = os.listdir(dir[gtype[type]])
    return files


#获取处理开始点，即从哪一个样本开始处理
def getStartPoint(type):
    ldata = 0
    start = -1
    datalist = os.listdir(save_dir)
    curlist = []
    for i in datalist:
        if i.startswith(type):
            curlist.append(i)
    if len(curlist) > 0:
        ldata = curlist[-1].split('-')[2].split('.')[0]
    return start if int(ldata) == 0 else int(ldata)


#数据预处理
def preProcessData(type,max_num):
    files = getapkpath(type)
    graphs = []
    start = getStartPoint(type)
    begin = start + 1
    for indx, f in enumerate(files):
        if indx > max_num:
            return
        if start != 0 and indx <= start:
            continue
        print("----  " + str(indx) + '  -------')
        fpath = os.path.join(dir[gtype[type]], f)
        if fpath.endswith('.sh'):
            with open('errlog.txt', 'a') as fi:
                fi.write(type + '\n')
                fi.write(str(indx) + '-----' + f + '---- not zip\n')
            continue
        print(fpath)
        try:
            cg, nfeatures = FCGextract(fpath)
        except:
            with open('errlog.txt', 'a') as fi:
                fi.write(type + '\n')
                fi.write(str(indx) + '-----' + f + '---- extract fail\n')
            continue
        if cg.order() == 0:
            with open('errlog.txt', 'a') as fi:
                fi.write(type + '\n')
                fi.write(str(indx) + '-----' + f + '---- no nodes\n')
            continue
        adj = nx.to_scipy_sparse_array(cg).tocoo()
        dgraph = dgl.from_scipy(adj)
        for key in nfeatures[0].keys():
            dgraph.ndata[key] = torch.tensor([i[key] for i in nfeatures])
        graphs.append(dgraph)
        if len(graphs) == 64 or indx == len(files) - 1:
            end = indx
            graphs_labels = {'glabel': torch.tensor([gtype[type] for i in range(len(graphs))])}
            savepath = os.path.join(save_dir, type + '-' + str(begin).zfill(4) + '-' + str(end).zfill(4) + '.bin')
            # if not os.path.exists(savepath):
            dgl.save_graphs(savepath, graphs,graphs_labels)
            graphs.clear()
            print('save: ' + str(begin) + '-' + str(end) + '.bin')
            begin = end + 1


if __name__ == '__main__':
    preProcessData('Benign', 1500)

