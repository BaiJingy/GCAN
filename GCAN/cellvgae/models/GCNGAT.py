
import torch  
from torch.nn import Sequential, Linear, ReLU, Dropout  
from torch_geometric.nn import GCNConv, GATConv, InnerProductDecoder  
from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops  
import torch.nn as nn  
import torch.nn.functional as F  
  
EPS = 1e-15  
MAX_LOGVAR = 10  
  
class GCNGAT(GAE):  
    def __init__(self, gcn_encoder, gat_encoder, decoder_nn_dim1=None, decoder=None, gcn_or_gat='GCNGAT'):  
        super(GCNGAT, self).__init__(None, decoder)  
        self.gcn_encoder = gcn_encoder  
        self.gat_encoder = gat_encoder  
        self.decoder = InnerProductDecoder() if decoder is None else decoder  
        self.decoder_nn_dim1 = decoder_nn_dim1  
        self.decoder_nn_dim2 = gcn_encoder.out_channels + gat_encoder.out_channels 
        self.decoder = MLPDecoder(input_dim=self.decoder_nn_dim2 * 2, hidden_dim=128, output_dim=14) 

        # self.dropout = Dropout(p=0.5)  # 初始化dropout层，p是dropout的概率
          
        if decoder_nn_dim1:  
            self.decoder_nn = Sequential(  
                Linear(in_features=self.decoder_nn_dim1, out_features=self.decoder_nn_dim2),  
                ReLU(),  
                Dropout(0.4),  
            )  
  
    def encode(self, x, edge_index):  
        gcn_z, _ = self.gcn_encoder(x, edge_index)  
        gat_z, _, _ = self.gat_encoder(x, edge_index)  
          
        assert gcn_z.size(1) == gat_z.size(1), "GCN and GAT output dimensions must match"  
          
        z = torch.cat([gcn_z, gat_z], dim=1)  
          
        batch_size, num_features = z.size()  
        self.__logvar__ = torch.zeros(batch_size, num_features).to(z.device)  
        self.__logvar__ = self.__logvar__.clamp(max=MAX_LOGVAR)  
          
        # Reparametrization  
        self.__mu__ = z  # 这里将z作为mu存储，假设在编码过程中没有进行其他修改  
        std = torch.exp(0.5 * self.__logvar__)  
        eps = torch.randn_like(self.__mu__)  
        self.__z_reparam__ = self.__mu__ + eps * std  
          
        return self.__z_reparam__  
  
    def kl_loss(self, mu=None, logvar=None):  
        mu = self.__mu__ if mu is None else mu  
        logvar = self.__logvar__ if logvar is None else logvar.clamp(max=MAX_LOGVAR)  
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))  
  
    # def recon_loss(self, pos_edge_index, neg_edge_index=None):  
    #             # 确保pos_edge_index是长整型  
    #     pos_edge_index = pos_edge_index.long()  
    #     z = self.__z_reparam__  # 使用重新参数化后的潜在表示  
    #     self.decoded = self.decoder(z, pos_edge_index)  
    #     pos_loss = -torch.log(self.decoded + EPS).mean()  
        
    #     # Do not include self-loops in negative samples  
    #     pos_edge_index, _ = remove_self_loops(pos_edge_index)  
    #     neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) if neg_edge_index is None else neg_edge_index  
    #     neg_edge_index, _ = add_self_loops(neg_edge_index)  
        
    #     self.decoded_neg = self.decoder(z, neg_edge_index)  
    #     neg_loss = -torch.log(1 - self.decoded_neg + EPS).mean()  

    #     return pos_loss + neg_loss
    def recon_loss(self, pos_edge_index, neg_edge_index=None):  
        # 确保pos_edge_index是长整型  
        pos_edge_index = pos_edge_index.long()  
        z = self.__z_reparam__  # 使用重新参数化后的潜在表示  
        self.decoded = self.decoder(z, pos_edge_index)  
        pos_loss = -torch.log(self.decoded + EPS).mean()  
        
        # Do not include self-loops in negative samples  
        # 注意这里只取元组的第一个元素  
        pos_edge_index_no_loops, _ = remove_self_loops(pos_edge_index)  
        
        # 如果neg_edge_index是None，则进行负采样，否则直接使用传入的neg_edge_index  
        if neg_edge_index is None:  
            neg_edge_index = negative_sampling(pos_edge_index_no_loops, z.size(0))  
            # 注意这里添加了自环到负样本中，同样只取元组的第一个元素  
            neg_edge_index_with_loops, _ = add_self_loops(neg_edge_index)  
        else:  
            neg_edge_index_with_loops = neg_edge_index  
        
        self.decoded_neg = self.decoder(z, neg_edge_index_with_loops)  
        neg_loss = -torch.log(1 - self.decoded_neg + EPS).mean()  
    
        return pos_loss + neg_loss
    

  
class MLPDecoder(nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim):  
        super(MLPDecoder, self).__init__()  
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
  
    def forward(self, z, edge_index):  
        # 假设z是节点的嵌入表示，edge_index是边的索引  
        # 首先，根据边的索引从z中获取节点的嵌入  
        # print(f"edge_index type: {type(edge_index)}, value: {edge_index}")
        edge_index1 = (edge_index[:, 1], edge_index[:, 2])
        row, col = edge_index1  
        z_row = z[row]  
        z_col = z[col]  
          
        # 拼接节点嵌入  
        concatenated = torch.cat([z_row, z_col], dim=1)  
          
        # 通过MLP得到解码后的结果  
        x = F.relu(self.fc1(concatenated))  
        decoded = torch.sigmoid(self.fc2(x))  # 使用sigmoid确保输出在0和1之间  
          
        return decoded  


# class MLPDecoder(nn.Module):    
#     def __init__(self, input_dim, hidden_dim, output_dim):    
#         super(MLPDecoder, self).__init__()    
#         self.fc1 = nn.Linear(input_dim, hidden_dim)    
#         self.fc2 = nn.Linear(hidden_dim, output_dim)  # 修改这里的output_dim  
    
#     def forward(self, z, edge_index):    
#         edge_index1 = (edge_index[:, 1], edge_index[:, 2])
#         row, col = edge_index1   
#         z_row = z[row]    
#         z_col = z[col]    
          
#         # 拼接节点嵌入  
#         concatenated = torch.cat([z_row, z_col], dim=1)    
          
#         # 通过MLP得到解码后的结果  
#         x = F.relu(self.fc1(concatenated))    
#         decoded = self.fc2(x)  # 去除sigmoid，因为聚类任务可能不需要限制输出范围  
          
#         return decoded
  
# 在您的GCNGAT类中  

