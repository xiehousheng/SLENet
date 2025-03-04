import ml_collections



def get_SLENet_config():
  
    
    ###
    config = ml_collections.ConfigDict()
    ###skip type
    config.if_selfskip = False
    config.if_crossskip = False
    config.if_crossaddselfskip = True
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
  
    config.depths_cam = (2, 2, 4, 2)
    config.num_heads_cam = (4, 4, 8, 8)
    config.window_size_cam = (5, 6, 7, 7)
    config.mlp_ratio_cam = 4
    config.qkv_bias_cam = False

  
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
  
    config.img_size = (352, 192, 96)
    config.resize = (352, 192, 96)
    
    config.deepsuper_similarity_weight = [0.25, 0.5, 1]
    config.deepsuper_smooth_weight = [0.25, 0.5, 1]
    config.deepsuper_jet_weight = [0.25, 0.5, 1]
    config.sup_level = [3,4,5] # [0,1,2,3,4,5] in which 0 represents the bottle level


    config.int_steps = 7
    config.is_diff = True
    config.lamda_smooth = 1
    config.lamda_jet = 3
    config.lamda_similarity = 1
    config.lossname_rigid = 'MSE'
    config.lossname_deform = 'MIND'
    config.device = '2'
   
    config.n_classes = 4
    config.rigid_class = 6
    config.com_initial = False

    config.modelname = 'SMILE_Hip_CBCT'
    config.if_crossam = False
    ## pure transformer decoder
    config.decode_depths = [2,4,2]
    config.decode_num_heads = [8,8,4]
    config.decode_window_size = [5,6,7]
    config.decode_depths_cam = [2,4,2]
    config.decode_num_heads_cam = [8,8,4]
    config.decode_window_size_cam = [5,6,7]
    return config


def get_3DTransMorphNoRelativePosEmbd_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphSin_config():
    '''
    TransMorph with Sinusoidal Positional Embedding
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = True
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.pos_embed_method = 'relative'
    return config

def get_3DTransMorphLrn_config():
    '''
    TransMorph with Learnable Positional Embedding
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = True
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphNoConvSkip_config():
    '''
    No skip connections from convolution layers

    Computational complexity:       577.34 GMac
    Number of parameters:           63.56 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.pos_embed_method = 'relative'
    return config

def get_3DTransMorphNoTransSkip_config():
    '''
    No skip connections from Transformer blocks

    Computational complexity:       639.93 GMac
    Number of parameters:           58.4 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphNoSkip_config():
    '''
    No skip connections

    Computational complexity:       639.93 GMac
    Number of parameters:           58.4 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphLarge_config():
    '''
    A Large TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (2, 2, 12, 2)
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphSmall_config():
    '''
    A Small TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 48
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 4, 4)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DTransMorphTiny_config():
    '''
    A Tiny TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 6
    config.depths = (2, 2, 4, 2)
    config.num_heads = (2, 2, 4, 4)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config
