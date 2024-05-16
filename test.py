from utils.utils_for_model import read_from_logs,create_parser


config_dict = {
    'aug': 'mix',
    'bsz': 64,
    'nb_nodes':50,
    'model_lr': 2e-5,
    'nb_batch_per_epoch': 300,
    'data_path':'./',
    'checkpoint_model': 'n',
    'aug_num': 16,
    'test_aug_num': 16,
    'num_state_encoder': 2,
    'dim_emb': 128,
    'dim_ff':512,
    'nb_heads': 8,
    'action_k': 15,
    'nb_layers_state_encoder': 2,
    'nb_layers_action_encoder': 2,
    'nb_layers_decoder': 3,
    'nb_epochs': 400,
    'problem': 'tsp',
    'gamma': 0.99,
    'dim_input_nodes': 2,
    'batchnorm':False,
    'gpu_id': 0,
    'loss_type':'n',
    'train_joint':'n',
    'nb_batch_eval': 80,
    'if_use_local_mask':False,
    'if_agg_whole_graph':False,
    'tol':1e-3,
}
state_k = [35,50,65]
custom_parser, args = create_parser(config_dict)
config = custom_parser.parse_args(namespace=args)

args = read_from_logs(args)
args.state_k = state_k[:args.num_state_encoder]
print(args)