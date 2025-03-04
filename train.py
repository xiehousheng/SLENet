from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import numpy as np
import os
import utils2
import time
import pandas as pd
from dataset import DataSet,TrainRandomSampler,TestRandomSampler
from utils import neg_Jdet_loss_sigmoid,generate_grid,DiceLong,DiceLong_seg,MulticlassDiceLossVectorize,read_csv_to_list,GradientDiffusionLoss,Grad,MutualInformation,BCELoss
import utils
from medpy.io import load, save
import torch.nn.functional as F
from xhs_loss import GradientDiffusionLoss, MSE, Grad, MIND_loss, MutualInformation,BCELoss
import os


from model import SLENet,Discriminator,CONFIGS,SpatialTransformer



class register_model2(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model2, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x, device):
        img = x[0]
        flow = x[1]
        out = self.spatial_trans(img, flow)
        return out

if __name__ == '__main__':
    wandb.init(project="abdo",name="slenet",    
        config={
        "x_axis": "epoch", 
        "smooth_factor": 0,
        "grouping": None
    })

    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_batch_size =1
    test_batch_size = 1
    validate_batch_size = 1
    learning_rate = 0.0001
    volume_size =[192, 160, 192]
    dataset_path_MR = './data_prepare/CTMR_Hip_114/MR_all_114'
    dataset_path_CT = './data_prepare/CTMR_Hip_114/CT_all_114'
 
    s, head = load('/data2/fuxian/dataset/AbdomenMRCT/imagesTr/AbdomenMRCT_0001_0000.nii.gz')
    patient_num = int(len(os.listdir(dataset_path_MR)) / 2) # total patients
    print('patient_num:', patient_num)
    test_num = round(patient_num * 0.2)
    valid_num = int(test_num * 0.5)
    print('test_num', test_num)
    print('valid_num', valid_num)
    
    train_num = patient_num - test_num
    # print(train_num)   21
    

    moving_vol_names_tr = read_csv_to_list('./mr_images_train.csv')
    print(moving_vol_names_tr)
    fixed_vol_names_tr = read_csv_to_list('./ct_images_train.csv')
    print(fixed_vol_names_tr)
    print('len(moving_vol_names_tr):', len(moving_vol_names_tr))  # 290*4 = 1160
    print('len(fixed_vol_names_tr):', len(fixed_vol_names_tr))

    moving_vol_names_te = read_csv_to_list('./mr_images_test.csv')
    fixed_vol_names_te = read_csv_to_list('./ct_images_test.csv')
    print('len(moving_vol_names_te):', len(moving_vol_names_te))  # 83*4 = 332
    print('len(fixed_vol_names_te):', len(fixed_vol_names_te))
    
    moving_vol_names_val = pd.read_csv('./moving_vol_names_val.csv', header=None)
    fixed_vol_names_val = pd.read_csv('./fixed_vol_names_val.csv', header=None)
    print('len(moving_vol_names_val):', len(moving_vol_names_val))  # 83* 4 = 332
    print('len(fixed_vol_names_val):', len(fixed_vol_names_val))



    # # # # obtain the according segmentations

    fixed_vol_names_tr_mask = read_csv_to_list('./ct_labels_train.csv')

    moving_vol_names_tr_mask = read_csv_to_list('./mr_labels_train.csv')


    fixed_vol_names_te_mask = read_csv_to_list('./ct_labels_test.csv')

    moving_vol_names_te_mask = read_csv_to_list('./mr_labels_test.csv')


    fixed_vol_names_val_mask = pd.read_csv('./fixed_vol_names_val_mask.csv', header=None)


    moving_vol_names_val_mask = pd.read_csv('./moving_vol_names_val_mask.csv', header=None)



    train_pair_num = len(moving_vol_names_tr)
    test_pair_num = len(moving_vol_names_te)
    val_pair_num = len(moving_vol_names_val)
    print('train_pair_num:', train_pair_num)
    print('test_pair_num:', test_pair_num)
    print('val_pair_num:', val_pair_num)

   

    config = CONFIGS['SLENet']
    model = SLENet(config)
    model.to(device)
    Discriminator=Discriminator()
    Discriminator.to(device)
    

    optimizer_gen = Adam(model.parameters(), lr=learning_rate)
    optimizer_dis = Adam(Discriminator.parameters(), lr=learning_rate)


    
    wandb.watch(model)
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(volume_size, 'nearest')
    reg_model.cuda()
    # reg_model_bilin = utils_TM.register_model(volume_size, 'bilinear')
    # reg_model_bilin.cuda()


    reg_model_bilin = register_model2(volume_size, 'bilinear')
    reg_model_bilin.cuda()

    reg_down_model = utils.register_model((48, 40, 48), 'nearest')
    reg_down_model.cuda()
    check = False

    if check:
        reload_path = ''
       
        model_data = torch.load(reload_path)
        model.load_state_dict(model_data['model'],strict=True)

        best_dice_val = 0
        best_dice_te = 0
        start_epochs = 0
        max_epochs = 2500
        model.to(device)
       
    else:
        best_dice_val = 0
        best_dice_te = 0
        start_epochs = 0
        max_epochs = 2500

    loss_similarity = utils2.MSE_nomask().loss
    loss_dice = utils2.Dice().loss
    Dicevolume_reg = DiceLong(num_clus = 5)
    Dicevolume = DiceLong_seg(num_clus = 2)
    loss_ncc = utils2.NCC().loss
    loss_L1 = utils2.L1().loss
    loss_nccmask = utils2.NCC_mask().loss
    loss_grad = utils2.Grad(penalty='l2').loss
    loss_diffusion = GradientDiffusionLoss(device = device, penalty='l2')
    loss_smooth = Grad('l2', loss_mult=1).loss
    loss_MI = MutualInformation(device=device)
    loss_MIND = MIND_loss(device)
    loss_bce=BCELoss()
    loss_ce=nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
    jet_lossfunc = neg_Jdet_loss_sigmoid
 
    dice_lossfunc = MulticlassDiceLossVectorize()

    
    weights = [1, 0.1]
    
    day_hour_minute = time.strftime("%d%H%M", time.localtime())
    strname = 'slenet'
    model_sub_dir = 'hip'
    savedatasubPath = "%s/%s" % (model_sub_dir, strname)
    if not os.path.exists(savedatasubPath):
        os.makedirs(savedatasubPath)
    model_savePath = "%s/%s/model" % (savedatasubPath, day_hour_minute)
    # *************#
    if not os.path.exists(model_savePath):
        os.makedirs(model_savePath)
    train_png_savePath = "%s/%s/train_png" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(train_png_savePath):
        os.makedirs(train_png_savePath)

    val_png_savePath = "%s/%s/val_png" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(val_png_savePath):
        os.makedirs(val_png_savePath)

    test_png_savePath = "%s/%s/tesst_png" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(test_png_savePath):
        os.makedirs(test_png_savePath)
    #*************#
    cpt_savePath_val = "%s/%s/cpt_val" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(cpt_savePath_val):
        os.makedirs(cpt_savePath_val)

    cpt_savePath_te = "%s/%s/cpt_te" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(cpt_savePath_te):
        os.makedirs(cpt_savePath_te)
    #*************#
    volume_savePath = "%s/%s/volume" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(volume_savePath):
        os.makedirs(volume_savePath)

    #*************#
    txt_savePath = "%s/%s/txt_log" % (savedatasubPath, day_hour_minute)
    if not os.path.exists(txt_savePath):
        os.makedirs(txt_savePath)

    for epoch in range(start_epochs, max_epochs):
        model.train()
    
        utils2.adjust_learning_rate_power(optimizer_gen, epoch, max_epochs, init_lr =  learning_rate, power=0.9)
        epoch_loss = [0, 0]
        vxm_dice_train = [0,0]
        vxm_mse_epoch_loss = [0, 0]
        t_s_distance = [0, 0]
  
        my_sampler = TrainRandomSampler(train_pair_num)
        my_batch_sampler = torch.utils.data.BatchSampler(my_sampler, batch_size=train_batch_size, drop_last=False)

       
    


        train_dataset = DataSet(fixed_vol_names_tr, moving_vol_names_tr,
                                             fixed_vol_names_tr_mask, moving_vol_names_tr_mask,aug=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=my_batch_sampler, num_workers = 1)
        train_dataloader = tqdm(train_dataloader)

        step_num=1
        
        # train
        for i, data in enumerate(train_dataloader):
          
            
            inmoving, inmoving_mask, infixed, infixed_mask, idx = \
                data[0].to(device), data[1].to(device),data[2].to(device), data[3].to(device),\
                list(data[4])[0]
            patientid = str(idx)
            
        
            x_in = torch.cat((inmoving,infixed), dim=0)  # (2,1,160,192,224)
            warp_moving, fix_img, out_velocity, out_disp, ct_seg, mr_seg,mr_feat, ct_seg_feat = model(x_in,inmoving_mask,infixed_mask)
            warp_moving, fix_img, out_velocity, out_disp=warp_moving[0], fix_img[0], out_velocity[0], out_disp[0]
            down_flow = F.interpolate(out_disp/4, size=mr_seg.size()[2:], mode='trilinear')
            mr_trans_label = reg_down_model([mr_seg, down_flow])


            x_label=torch.where(infixed_mask!=0,1,0).float()
            x_warp_label=torch.where(mr_seg>0.3,1,0).float()


            x_label=F.interpolate(x_label, size=infixed_mask.size()[2:], mode='trilinear')
            x_label = nn.functional.one_hot(x_label.long(), num_classes=2)
            x_label = torch.squeeze(x_label, 1)
            x_label = x_label.permute(0, 4, 1, 2, 3).contiguous().float() # 1 36 160 192 224
            x_warp_label=F.interpolate(x_warp_label, size=infixed_mask.size()[2:], mode='trilinear')
            x_warp_label = nn.functional.one_hot(x_warp_label.long(), num_classes= 2)
            x_warp_label = torch.squeeze(x_warp_label, 1)
            x_warp_label = x_warp_label.permute(0, 4, 1, 2, 3).contiguous().float() # 1 36 160 192 224
            x_warp_label=reg_model_bilin([x_warp_label.float(),out_disp.float()],device)

            
            vxm_mi_loss = loss_MIND(warp_moving, fix_img)
            smooth_loss = loss_diffusion(infixed, out_velocity)


            downsampled_size = ct_seg.size()[2:]
            

         
            infixed_mask = F.interpolate(infixed_mask, size=downsampled_size, mode='trilinear')
            seg_gt = torch.where(infixed_mask!=0,1,0)
            seg_loss = loss_bce(ct_seg, seg_gt)
            output_seg_bce=loss_bce(mr_trans_label.float(),seg_gt.detach())
        
            reg_loss = vxm_mi_loss  + smooth_loss*0.1
        
            grid = generate_grid( (192, 160, 192))
            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
            jetloss=jet_lossfunc(out_disp, grid)
           
    
            noise=F.relu(seg_gt-mr_trans_label)
            noisemrseg=noise+mr_trans_label

        


            outs0 = Discriminator(noisemrseg.float())
            outs1 = Discriminator(seg_gt.float())
            d_loss=((outs0) ** 2)+((outs1 - 1) ** 2)

            optimizer_dis.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_dis.step()
            deform = dice_lossfunc(x_warp_label, x_label)

            outs0 = Discriminator(noisemrseg.float())
            g_loss=(outs0 - 1) ** 2+output_seg_bce*0.5+deform*0.1

            
            



            if epoch>80:
                batchloss = seg_loss*0.5+reg_loss*5+jetloss*10
            else:
                batchloss = seg_loss*0.5+reg_loss*5

            optimizer_gen.zero_grad()
            batchloss.backward()
            optimizer_gen.step()


        
            step_num=step_num+1



            batchloss=reg_loss

            epoch_loss[0] += len(data[0])
            vxm_mse_epoch_loss[0] += len(data[0])
            epoch_loss[1] += batchloss.detach().cpu().numpy() * len(data[0])
            vxm_mse_epoch_loss[1] += batchloss.detach().cpu().numpy() * len(data[0])


            if i>100 and i%500==0:

        
                train_epoch_loss = epoch_loss[1] / epoch_loss[0]
                vxm_epoch_loss_mse = vxm_mse_epoch_loss[1]/vxm_mse_epoch_loss[0]
                train_dataloader.set_description("Train loss %f" % (train_epoch_loss))
                
            
                # evaluate on validate dataset
                model.eval()
                v_epoch_loss = [0, 0]
                vxm_dice_validate = [0, 0]
                v_vxm_mse_epoch_loss = [0, 0]
                reg_dice_validate=[0,0]


                val_epoch_dice = 0
                reg_val_epoch_dice = 0

                te_epoch_loss = [0, 0]
                vxm_dice_te = [0, 0]
                reg_dice_test=[0,0]
                mr_dice_te=[0,0]
                ct_dice_te=[0,0]
                dice_0= [0, 0]
                dice_1= [0, 0]
                dice_2= [0, 0]
                v_my_sampler = TestRandomSampler(test_pair_num)
                v_my_batch_sampler = torch.utils.data.BatchSampler(v_my_sampler, batch_size=validate_batch_size, drop_last=False)
                te_dataset = DataSet(fixed_vol_names_te,moving_vol_names_te, 
                                                        fixed_vol_names_te_mask, moving_vol_names_te_mask,aug=False)
                te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_sampler=v_my_batch_sampler, num_workers = 1)
                te_dataloader = tqdm(te_dataloader)
                # validate
                with torch.no_grad():
                    for i, data in enumerate(te_dataloader):
                        inmoving, inmoving_mask, infixed, infixed_mask, idx = \
                        data[0].to(device), data[1].to(device),data[2].to(device), data[3].to(device),\
                        list(data[4])[0]
                        patientid = str(idx)
                         
                        x_in = torch.cat((inmoving,infixed), dim=0)  # (2,1,160,192,224)
                        warp_moving, fix_img, out_velocity, out_disp, ct_seg, mr_seg ,mr_feat, ct_seg_feat= model(x_in,inmoving_mask,infixed_mask)
                        warp_moving, fix_img, out_velocity, out_disp=warp_moving[0], fix_img[0], out_velocity[0], out_disp[0]

                       

                    
                        source_mask = F.one_hot(inmoving_mask.long(), num_classes=5)
                        source_mask = torch.squeeze(source_mask, 1)
                        source_mask = source_mask.permute(0, 4, 1, 2, 3).contiguous() # 1 4 160 192 224

                        x_segs = []
                        for i in range(5):
                            def_seg = reg_model_bilin([source_mask[:, i:i + 1, ...].float(), out_disp.float()], device)
                            x_segs.append(def_seg)
                        x_segs = torch.cat(x_segs, dim=1)
                        def_out = torch.argmax(x_segs, dim=1, keepdim=True)

                        vxm_dice = Dicevolume_reg(def_out.long().to(device), infixed_mask.long().to(device))

                       
                        mean_test_dice = torch.mean(vxm_dice)



                      
                    
                        print('test vxm_dice:',mean_test_dice)
                        with open('%s/test_metric_after.txt'%(txt_savePath), 'a') as f:
                            f.writelines('patient: {}, DCe:{} \n'.format(patientid, mean_test_dice))

                        vxm_dice_te[0] += len(data[0])
                        vxm_dice_te[1] += mean_test_dice.detach().cpu().numpy() * len(data[0])

                        downsampled_size = ct_seg.size()[2:]

                        # 使用三线性插值进行下采样
                        infixed_mask = F.interpolate(infixed_mask, size=downsampled_size, mode='trilinear')
                        inmoving_mask = F.interpolate(inmoving_mask, size=downsampled_size, mode='trilinear')
                
                        # 
                        seg_gt = torch.where(infixed_mask!=0,1,0)
                        mr_seg_gt = torch.where(inmoving_mask!=0,1,0)
                        ct_seg=torch.where(ct_seg>0.3,1,0)
                        mr_seg=torch.where(mr_seg>0.3,1,0)
                        ct_dice = Dicevolume(ct_seg.long().to(device), seg_gt.long().to(device))
                        mr_dice = Dicevolume(mr_seg.long().to(device), mr_seg_gt.long().to(device))

                        mr_dice_te[0] += len(data[0])
                        mr_dice_te[1] += mr_dice.detach().cpu().numpy() * len(data[0])
                        ct_dice_te[0] += len(data[0])
                        ct_dice_te[1] += ct_dice.detach().cpu().numpy() * len(data[0])

                    mr_te_epoch_dice=mr_dice_te[1]/mr_dice_te[0]
                    ct_te_epoch_dice=ct_dice_te[1]/ct_dice_te[0]
                
                    te_epoch_dice = vxm_dice_te[1] / vxm_dice_te[0]
                    print("***********************************************",te_epoch_dice)
                
                

                    wandb.log({"train_epoch_loss": train_epoch_loss, 'test_reg_Dice': te_epoch_dice,'mr_seg_dice':mr_te_epoch_dice,'ct_seg_dice':ct_te_epoch_dice}) 


            
                
                if epoch>-1:
        
                    if te_epoch_dice > best_dice_te:
                        best_dice_te = te_epoch_dice
                        utils2.checkpointdice(model, optimizer_gen, te_epoch_dice, epoch, cpt_savePath_te)
