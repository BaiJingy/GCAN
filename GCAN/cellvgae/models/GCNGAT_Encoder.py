import torch  
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv  
from torch import nn  
import torch.nn.functional as F

from functools import partial
  
class GCNGAT_Encoder(nn.Module):  
    def __init__(self, num_hidden_layers, num_heads, in_channels, hidden_dims, latent_dim, dropout, concat, use_gcn_first=True, v2=False):  
        super(GCNGAT_Encoder, self).__init__()  
        assert (num_hidden_layers in [2, 3]), 'The number of hidden layers must be 2 or 3.'  
        assert (num_hidden_layers == len(hidden_dims)), 'The number of hidden layers must match the number of hidden output dimensions.'  
  
        self.in_channels = in_channels  
        self.latent_dim = latent_dim  
        self.num_heads = num_heads  
        self.num_hidden_layers = num_hidden_layers  
        self.use_gcn_first = use_gcn_first  # Flag to indicate whether to use GCN in the first layer  
        self.v2 = v2  
  
        # Define the convolution type based on v2 flag  
        self.conv = GATv2Conv if v2 else GATConv  
  
        # If using GCN in the first layer  
        if self.use_gcn_first:  
            self.gcn_layer1 = GCNConv(self.in_channels, hidden_dims[0])  
            # The input dimension for the first GAT layer is the output dimension of GCN  
            in_dim_gat1 = hidden_dims[0] #原来 
            #调整
            # in_dim_gat1 = self.in_channels /2     
        else:  
            # If not using GCN, the input dimension for the first GAT layer is the original feature dimension  
            in_dim_gat1 = self.in_channels  
  
        # First GAT layer  
        self.hidden_layer1 = self.conv(  
            in_channels=in_dim_gat1, out_channels=hidden_dims[0],  
            heads=self.num_heads['first'] if isinstance(self.num_heads, dict) else self.num_heads,  
            dropout=dropout[0],  
            concat=concat['first'] if isinstance(concat, dict) else concat  
        )  
  
        # Adjust input dimension for the next layer based on whether concat is True  
        in_dim2 = hidden_dims[0] * self.num_heads['first'] if concat['first'] else hidden_dims[0]  
  
        # Second GAT layer  
        self.hidden_layer2 = self.conv(  
            in_channels=in_dim2, out_channels=hidden_dims[1],  
            heads=self.num_heads['second'] if isinstance(self.num_heads, dict) else self.num_heads,  
            dropout=dropout[1],  
            concat=concat['second'] if isinstance(concat, dict) else concat  
        )  
  
        # Third GAT layer (optional)  
        self.hidden_layer3 = None  
        if num_hidden_layers == 3:  
            in_dim3 = hidden_dims[1] * self.num_heads['second'] if concat['second'] else hidden_dims[1]  
            self.hidden_layer3 = self.conv(  
                in_channels=in_dim3, out_channels=hidden_dims[2],  
                heads=self.num_heads['third'],
                dropout=dropout[2],  
                concat=concat['third'] 
            )  
  
        if num_hidden_layers == 2:
            in_dim_final = hidden_dims[-1] * self.num_heads['second'] if concat['second'] else hidden_dims[-1]
        elif num_hidden_layers == 3:
            in_dim_final = hidden_dims[-1] * self.num_heads['third'] if concat['third'] else hidden_dims[-1]
        self.conv_mean = self.conv(in_channels=in_dim_final, out_channels=latent_dim,
                                   heads=self.num_heads['mean'], concat=False, dropout=0.2)
        self.conv_log_std = self.conv(in_channels=in_dim_final, out_channels=latent_dim,
                                      heads=self.num_heads['std'], concat=False, dropout=0.2)
  
    
    def forward(self, x, edge_index):  
        # First layer: GCN or GAT  
        if self.use_gcn_first:  
            # _, attn_w_1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)  
            hidden_out0 = F.relu(self.gcn_layer1(x, edge_index))  
        else:  
            hidden_out0, attn_w_0 = self.hidden_layer1(x, edge_index, return_attention_weights=True)  
            hidden_out0 = F.relu(hidden_out0)  
        
        hidden_out1, attn_w_1 = self.hidden_layer1(hidden_out0, edge_index, return_attention_weights=True)
        hidden_out1 = F.relu(hidden_out1) 
        hidden_out1 = F.dropout(hidden_out1, p=0.4, training=self.training)         
  
        # Second layer: GAT  
        hidden_out2, attn_w_2 = self.hidden_layer2(hidden_out1, edge_index, return_attention_weights=True)  
        hidden_out2 = F.relu(hidden_out2)  
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)  
  
        hidden_out3, attn_w_3 = None, None
        if self.hidden_layer3:
            hidden_out3, attn_w_3 = self.hidden_layer3(hidden_out2, edge_index, return_attention_weights=True)
        last_out = hidden_out3 if self.hidden_layer3 else hidden_out2
        z_mean, attn_w_mean = self.conv_mean(last_out, edge_index, return_attention_weights=True)
        z_log_std, attn_w_log_std = self.conv_log_std(last_out, edge_index, return_attention_weights=True)

        if self.hidden_layer3:
            return z_mean, z_log_std, (attn_w_1, attn_w_2, attn_w_3, attn_w_mean, attn_w_log_std)
        return z_mean, z_log_std, (attn_w_1, attn_w_2, attn_w_mean, attn_w_log_std) 

