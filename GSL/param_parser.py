import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="GSL", help='Model to use for training')

    parser.add_argument("--dataset_name", type=str, default="IMDB-BINARY", help="Folder with training graph jsons.")

    parser.add_argument("--graph_type", type=str, default="epsilonNN", help="epsilonNN, KNN, prob")

    parser.add_argument("--graph_metric_type", type=str, default="dot")

    parser.add_argument("--repar", type=bool, default=True, help="Default is True.")

    parser.add_argument("--num_layers", type=int, default=2, help="Default is 2.")

    parser.add_argument('--num_classes',type=int, default=2, help='number of class')

    parser.add_argument("--hidden_dim", type=int, default=8, help="Default is 8.")

    parser.add_argument("--folds", type=int, default=10, help="Default is 10.")

    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs. Default is 200.")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default is 0.001.")

    parser.add_argument('--lr_decay_factor', type=float, default=0.5)

    parser.add_argument('--lr_decay_step_size', type=int, default=50)

    parser.add_argument("--weight_decay", type=float, default=0.00005, help="Adam weight decay. Default is 5*10^-5.")

    parser.add_argument("--batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--test_batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--beta", type=float, default=0.1, help="Default is 0.1")

    parser.add_argument("--IB_size", type=int, default=256, help="Default is 16.")

    parser.add_argument("--num_per", type=int, default=16, help="Default is 16")

    parser.add_argument("--feature_denoise", type=bool, default=False, help="Default is False.")

    parser.add_argument("--set_epsilon", type=float, default=0.8, help="Default is 0.2.")

    parser.add_argument("--graph_skip_conn", type=float, default=0.0, help="Default is 0.0.")

    parser.add_argument("--graph_include_self", type=bool, default=True, help="Default is True.")

    parser.add_argument('--PATH', type=str, default="data_origin", help="dataset")

    parser.add_argument('--readout', type=str, default="mean", help="readout")

    parser.add_argument('--tau', type=float, default=0.2, help="Default is 0.2")

    parser.add_argument('--average', type=bool, default=False)

    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')

    parser.add_argument('--lamb', type=float, default=0.0001, help='motif parameter')

    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

    parser.add_argument('--conv_name', type=str, default='ARMAConv', help='conv')

    parser.add_argument('--pool_name', type=str, default='EdgePooling', help='pooling')

    parser.add_argument('--ignore_edge', type=bool, default=False, help='whether ignore edge')

    parser.add_argument('--fieldID_embedding', type=int, default=0, help='embedding fieldID')

    parser.add_argument('--train_all', type=bool, default=False, help='train model use all available field')

    parser.add_argument('--ignore_node_attr', type=bool, default=False, help='whether ignore node attribute')


    return parser.parse_args()
