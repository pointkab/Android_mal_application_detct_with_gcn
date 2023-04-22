import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import torch.nn as nn
import dgl

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim,allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim,allow_zero_in_degree=True)
        self.conv3 = dglnn.GraphConv(hidden_dim,hidden_dim,allow_zero_in_degree=True)
        self.conv4 = dglnn.GraphConv(hidden_dim,hidden_dim,allow_zero_in_degree=True)
        self.conv5 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # 应用图卷积和激活函数
        h = F.relu(self.conv1(g,h))
        h = F.relu(self.conv2(g,h))
        h = F.relu(self.conv3(g,h))
        h = F.relu(self.conv4(g, h))
        h = F.relu(self.conv5(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # 使用平均读出计算图表示
            hg = dgl.sum_nodes(g, 'h')
            return self.classify(hg)