# Requirements   需求
Installing GCAN with pip will attempt to install PyTorch and PyTorch Geometric, however it is recommended that the appropriate GPU/CPU versions are installed manually beforehand. For Linux:使用pip安装GCAN将尝试安装PyTorch和PyTorch Geometric，但建议事先手动安装适当的GPU/CPU版本。Linux:
1.	Install PyTorch GPU:   1.	安装PyTorch GPU：
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
or PyTorch CPU:   或PyTorch CPU:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
2.	Install PyTorch Geometric:
conda install pyg -c pyg -c conda-forge
3.	(Optional) Install Faiss CPU:
conda install -c pytorch faiss-cpu
Faiss is only required if using the option --graph_type "KNN Faiss" . It is a soft dependency as it is not available for some platforms (currently Apple M1). Attempting to use CellVGAE with Faiss without installing it will result in an exception.
A GPU version of Faiss for CUDA 11.1 is not yet available.目前还没有针对CUDA 11.1的Faiss GPU版本。
4.	Install CellVGAE with pip:
pip install cellvgae --pre

# Example use（Enter the following command in the terminal to run it）示例用法（在终端中输入以下命令运行它）
python -m cellvgae --input_gene_expression_path "data_h5ad/mouse_bladder_cell.h5ad" --hvg 5000 --khvg 250 --graph_type "KNN Scanpy" --k 10 --graph_metric "euclidean" --save_graph --graph_convolution "GCNGAT" --num_hidden_layers 2 --hidden_dims 256 256 --num_heads 10 10 10 10 --dropout 0.4 0.4 0.4 0.4 --latent_dim 50 --epochs 200 --model_save_path "model_saved_out"  --umap –hdbscanPython -m cellvgae——input_gene_expression_path "data_h5ad/mouse_bladder_cell. Pythonh5ad“——hvg 5000——khvg 250——graph_type ”KNN Scanpy“——k10——graph_metric ”euclidean“——save_graph——graph_convolution ”GCNGAT“——num_hidden_layers 2——hidden_dims 256 256——num_heads 10 10 10 10 10——dropout 0.4 0.4 0.4 0.4 0.4——latent_dim 50——epochs 200——model_save_path ”model_saved_out"——umap -hdbscan

# Usage   使用
Invoke the training script with python -m cellvgae with the arguments detailed below:
usage: train [-h] [--input_gene_expression_path INPUT_GENE_EXPRESSION_PATH] [--hvg HVG] [--khvg KHVG] [--graph_type {KNN Scanpy,KNN Faiss,PKNN}] [--k K] [--graph_n_pcs GRAPH_N_PCS]
             [--graph_metric {euclidean,manhattan,cosine}] [--graph_distance_cutoff_num_stds GRAPH_DISTANCE_CUTOFF_NUM_STDS] [--save_graph] [--raw_counts] [--faiss_gpu]
             [--hvg_file_path HVG_FILE_PATH] [--khvg_file_path KHVG_FILE_PATH] [--graph_file_path GRAPH_FILE_PATH] [--graph_convolution {GAT,GATv2,GCN}] [--num_hidden_layers {2,3}]
             [--num_heads [NUM_HEADS [NUM_HEADS ...]]] [--hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]] [--dropout [DROPOUT [DROPOUT ...]]] [--latent_dim LATENT_DIM] [--loss {kl,mmd}] [--lr LR]
             [--epochs EPOCHS] [--val_split VAL_SPLIT] [--test_split TEST_SPLIT] [--transpose_input] [--use_linear_decoder] [--decoder_nn_dim1 DECODER_NN_DIM1] [--name NAME] --model_save_path MODEL_SAVE_PATH [--umap] [--hdbscan]
Train CellVGAE.
optional arguments:
  -h, --help            show this help message and exit
  --input_gene_expression_path INPUT_GENE_EXPRESSION_PATH
                        Input gene expression file path.
  --hvg HVG             Number of HVGs.
  --khvg KHVG           Number of KHVGs.
  --graph_type {KNN Scanpy,KNN Faiss,PKNN}--graph_type {KNN Scanpy，KNN Faiss，PKNN}
                        Type of graph.   图的类型。
  --k K                 K for KNN or Pearson (PKNN) graph.——k k k对于KNN或Pearson （PKNN）图。
  --graph_n_pcs GRAPH_N_PCS——graph_n_pcs graph_n_pcs
                        Use this many Principal Components for the KNN (only Scanpy).对KNN使用这么多主组件（只有Scanpy）。
  --graph_metric {euclidean,manhattan,cosine}——graph_metric{欧几里得、曼哈顿、余弦}
  --graph_distance_cutoff_num_stds GRAPH_DISTANCE_CUTOFF_NUM_STDS——graph_distance_cutoff_num_stds graph_distance_cutoff_num_stds
                        Number of standard deviations to add to the mean of distances/correlation values. Can be negative.与距离/相关值的平均值相加的标准差数。可以是负的。
  --save_graph          Save the generated graph to the output path specified by --model_save_path.——save_graph保存生成的图形到——model_save_path指定的输出路径。
  --raw_counts          Enable preprocessing recipe for raw counts.——raw_counts为原始计数启用预处理配方。
  --faiss_gpu           Use Faiss on the GPU (only for KNN Faiss).在GPU上使用Faiss（仅用于KNN Faiss）。
  --hvg_file_path HVG_FILE_PATH——hvg_file_path hvg_file_path
                        HVG file if not using command line options to generate it.HVG文件，如果不使用命令行选项生成它。
  --khvg_file_path KHVG_FILE_PATH——khvg_file_path khvg_file_path
                        KHVG file if not using command line options to generate it. Can be the same file as --hvg_file_path if HVG = KHVG.如果不使用命令行选项来生成它的话。如果HVG = KHVG，可以与——hvg_file_path相同。
  --graph_file_path GRAPH_FILE_PATH——graph_file_path graph_file_path
                        Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma), if not using command line options to generate it.指定为边列表的图形（每行一条边，节点由空格分隔，而不是逗号），如果不使用命令行选项来生成它。
  --graph_convolution {GCAN}——graph_convolution {GCAN}
  --num_hidden_layers {2,3}——num_hidden_layers {2,3}
                        Number of hidden layers (must be 2 or 3).隐藏层数（必须是2或3）。
  --num_heads [NUM_HEADS [NUM_HEADS ...]]
                        Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.每一层的注意头数。输入是一个长度必须匹配层总数= num_hidden_layers 2的列表。
  --hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]
                        Output dimension for each hidden layer. Input is a list that matches --num_hidden_layers in length.每个隐藏层的输出维度。输入是一个长度匹配——num_hidden_layers的列表。
  --dropout [DROPOUT [DROPOUT ...]]—dropout [dropout [dropout…   辍学……]]
  --latent_dim LATENT_DIM   ——latent_dim latent_dim——latent_dim——latent_dim——latent_dim——latent_dim latent_dim——latent_dimlatent_dim -latent_dim -latent_dim -latent_dim -latent_dim -latent_dim -latent_dim -latent_dim latent_dim
                        Latent dimension (output dimension for node embeddings).潜在维度（节点嵌入的输出维度）。
  --loss {kl,mmd}       Loss function (KL or MMD).——loss {kl，mmd}损失函数（kl或mmd）。
  --lr LR               Learning rate for Adam.亚当的学习率。
  --epochs EPOCHS       Number of training epochs.——epochs epochs训练周期数。
  --val_split VAL_SPLIT
                        Validation split e.g. 0.1.验证分割，例如0.1。
  --test_split TEST_SPLIT   ——test_split test_split
                        Test split e.g. 0.1.   测试分割，例如0.1。
  --transpose_input     Specify if inputs should be transposed.——transpose_input指定输入是否应该调换。
  --use_linear_decoder  Turn on a neural network decoder, similar to traditional VAEs.—use_linear_decoder打开一个神经网络解码器，类似于传统的vae。
  --decoder_nn_dim1 DECODER_NN_DIM1——decoder_nn_dim1 decoder_nn_dim1
                        First hidden dimenson for the neural network decoder, if specified using --use_linear_decoder.神经网络解码器的第一个隐藏维度，如果使用——use_linear_decoder指定。
  --name NAME           Name used for the written output files.——name name写入输出文件的名称。
  --model_save_path MODEL_SAVE_PATH——model_save_path model_save_path
                        Path to save PyTorch model and output files. Will create the entire path if necessary.PyTorch模型和输出文件的保存路径。如果需要，将创建整个路径。
  --umap                Compute and save the 2D UMAP embeddings of the output node features.计算并保存输出节点特征的2D umap嵌入。
  --hdbscan             Compute and save different HDBSCAN clusterings.——hdbscan计算并保存不同的hdbscan集群。

