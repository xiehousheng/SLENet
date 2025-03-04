import torch
import numpy as np
from medpy.io import load, save
import numpy as np

class DataSet(torch.utils.data.Dataset):


    def __init__(self, fixed_vol_names_tr, moving_vol_names_tr,fixed_vol_names_tr_mask, moving_vol_names_tr_mask, if_poly = False, aug = False):
        super(DataSet, self).__init__()
        self.moving_vol_names = moving_vol_names_tr
        self.fixed_vol_names = fixed_vol_names_tr
        self.moving_vol_mask_names = moving_vol_names_tr_mask
        self.fixed_vol_mask_names = fixed_vol_names_tr_mask
        self.files = []
        self.maskfiles = []
     
        self.weightpath_fixed = '/data2/dataset/AbdomenMRCT/imagesTr/'
        self.weightpath_moving = '/data2/dataset/AbdomenMRCT/imagesTr'
     



        for m_name, f_name in zip(self.moving_vol_names, self.fixed_vol_names):
            self.files.append({
                "Moving_image": m_name,
                "Fixed_image": f_name,
            })
      
            
          
        for mm_name, fm_name in zip(self.moving_vol_mask_names, self.fixed_vol_mask_names):
            self.maskfiles.append({
                "Moving_image_mask": mm_name,
                "Fixed_image_mask": fm_name,
            })
         
            

    
     

        self.transform_intensity = get_intensity_transform()            
      
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        datafiles = self.files[idx]
        maskfiles = self.maskfiles[idx]

        inmoving,head = load(datafiles['Moving_image'])

      
        inmoving = np.clip(inmoving, -1024, 1024)
        inmoving = (inmoving - inmoving.min())/(inmoving.max() - inmoving.min())
    
        inmoving = torch.from_numpy(inmoving).float().unsqueeze(0) 
       
        moving_mask,_ = load(maskfiles['Moving_image_mask'])
        moving_mask = torch.from_numpy(moving_mask.astype(float)).float().unsqueeze(0)
    
       
        infixed,_ = load(datafiles['Fixed_image'])
        


        infixed = np.clip(infixed, -300, 300)
        infixed -= infixed.min()
        infixed /= infixed.max()


        infixed = torch.from_numpy(infixed).float().unsqueeze(0)
        fixed_mask,_ =load(maskfiles['Fixed_image_mask'])
          
        fixed_mask = torch.from_numpy(fixed_mask).float().unsqueeze(0)
    
    
   
        dataidx = str(idx)

     
               
        return inmoving, moving_mask, infixed, fixed_mask, dataidx



class TrainRandomSampler(torch.utils.data.Sampler):
    """
    note: we should shuffle the train dataset every epoch
    """

    def __init__(self, train_num):
        train_index = [i for i in range(train_num)]
        random.shuffle(train_index)
        #print('index_order:', train_index)
        self.data_source = train_index

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


class TestRandomSampler(torch.utils.data.Sampler):
    """
    note: do not shuffle the test dataset every epoch
    """

    def __init__(self, test_num):
        test_index = [i for i in range(test_num)]
        self.data_source = test_index

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)