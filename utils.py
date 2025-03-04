import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class GradientDiffusionLoss(nn.Module):
    def __init__(self):
        super(GradientDiffusionLoss, self).__init__()
        
    def forward(self, fixed, velocity):
        dx = torch.abs(velocity[:, :, 1:, :, :] - velocity[:, :, :-1, :, :])
        dy = torch.abs(velocity[:, :, :, 1:, :] - velocity[:, :, :, :-1, :])
        dz = torch.abs(velocity[:, :, :, :, 1:] - velocity[:, :, :, :, :-1])

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        
    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        return 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

class Grad(nn.Module):
    def __init__(self):
        super(Grad, self).__init__()
        
    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        return torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

class MIND_loss(nn.Module):
    def __init__(self, win=7):
        super(MIND_loss, self).__init__()
        self.win = win
        
    def pdist_squared(self, x):
        xx = (x**2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # See paper: 
        # "MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration"
      
        
        kernel_size = radius * 2 + 1
        six_neighbourhood = torch.Tensor([[0,1,1],[1,1,0],[1,0,1],[1,1,2],[2,1,1],[1,2,1]]).long()
        
        # Create patches
        patches = F.unfold(img.reshape(img.shape[0], 1, img.shape[1], img.shape[2]*img.shape[3]), 
                          kernel_size=kernel_size, dilation=dilation, padding=radius*dilation)
        
        # Reshape to [B, C, H, W, D, patch_size]
        patches = patches.reshape(img.shape[0], -1, img.shape[1], img.shape[2], img.shape[3])
        
        # Compute patch features
        patch_indices = torch.zeros(six_neighbourhood.shape[0], kernel_size**2, dtype=torch.long, device=img.device)
        for i in range(six_neighbourhood.shape[0]):
            patch_indices[i] = torch.ravel_multi_index((six_neighbourhood[i, 0], six_neighbourhood[i, 1], six_neighbourhood[i, 2]), 
                                                       (kernel_size, kernel_size, kernel_size))
        
        patch_features = patches[:, :, patch_indices, :, :]
        
        # Compute MIND descriptor
        mind = torch.exp(-patch_features**2 / (0.1 * torch.mean(patch_features**2) + 1e-5))
        
        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true))**2)

class MutualInformation(nn.Module):
    def __init__(self, sigma_ratio=1, minval=0.0, maxval=1.0, num_bins=32):
        super(MutualInformation, self).__init__()
        self.sigma_ratio = sigma_ratio
        self.num_bins = num_bins
        self.minval = minval
        self.maxval = maxval
        
    def forward(self, y_pred, y_true):
        # Compute joint histogram
        nbins = self.num_bins
        
        # Scale data to [0, num_bins-1]
        y_pred = (y_pred - self.minval) / (self.maxval - self.minval) * (nbins - 1)
        y_true = (y_true - self.minval) / (self.maxval - self.minval) * (nbins - 1)
        
        # Clip values
        y_pred = torch.clamp(y_pred, 0, nbins-1)
        y_true = torch.clamp(y_true, 0, nbins-1)
        
        # Create soft histogram
        sigma = self.sigma_ratio * (nbins - 1) / 2
        
        # Create bin centers
        bin_centers = torch.linspace(0, nbins-1, nbins).to(y_pred.device)
        
        # Reshape inputs
        y_pred_flat = y_pred.reshape(-1, 1)
        y_true_flat = y_true.reshape(-1, 1)
        
        # Compute Gaussian kernel
        pred_kernel = torch.exp(-((y_pred_flat - bin_centers)**2) / (2 * sigma**2))
        true_kernel = torch.exp(-((y_true_flat - bin_centers)**2) / (2 * sigma**2))
        
        # Normalize kernels
        pred_kernel = pred_kernel / torch.sum(pred_kernel, dim=1, keepdim=True)
        true_kernel = true_kernel / torch.sum(true_kernel, dim=1, keepdim=True)
        
        # Compute joint histogram
        joint_hist = torch.matmul(pred_kernel.t(), true_kernel)
        joint_hist = joint_hist / torch.sum(joint_hist)
        
        # Compute marginal histograms
        pred_hist = torch.sum(joint_hist, dim=1)
        true_hist = torch.sum(joint_hist, dim=0)
        
        # Compute entropy
        pred_entropy = -torch.sum(pred_hist * torch.log2(pred_hist + 1e-7))
        true_entropy = -torch.sum(true_hist * torch.log2(true_hist + 1e-7))
        
        # Compute joint entropy
        joint_entropy = -torch.sum(joint_hist * torch.log2(joint_hist + 1e-7))
        
        # Compute mutual information
        mutual_info = pred_entropy + true_entropy - joint_entropy
        
        return -mutual_info  # Negative because we want to maximize MI
def neg_Jdet_loss_sigmoid(y_pred, sample_grid):
    y_pred = y_pred.permute(0, 2, 3, 4, 1)
    Jdet = JacboianDet(y_pred, sample_grid)
    selected_pos_Jdet = F.relu(Jdet)*1000
    selected_pos_Jdet_num = (torch.sigmoid(selected_pos_Jdet) - 0.5) * 2
    return 1 - torch.mean(selected_pos_Jdet_num)
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet
def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid

def context_mask(mask, kernel_size=3):
    """
    Create a context mask by dilating the input mask
    """
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size).to(mask.device)
    dilated_mask = F.conv3d(mask, kernel, padding=padding) > 0
    context_mask = dilated_mask.float() - mask
    return context_mask

def distillation_loss(student_features, teacher_features, temperature=1.0):
    """
    Compute knowledge distillation loss between student and teacher features
    """
    return F.kl_div(
        F.log_softmax(student_features / temperature, dim=1),
        F.softmax(teacher_features / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

def dice_assd(pred, target, smooth=1e-5):
    """
    Compute Dice score and Average Symmetric Surface Distance
    """
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    # Simple approximation of ASSD using boundary differences
    pred_boundary = get_boundary(pred)
    target_boundary = get_boundary(target)
    
    # Mean distance from pred to target
    pred_to_target = ((pred_boundary - target_boundary) ** 2).sum() / (pred_boundary.sum() + 1e-5)
    
    # Mean distance from target to pred
    target_to_pred = ((target_boundary - pred_boundary) ** 2).sum() / (target_boundary.sum() + 1e-5)
    
    assd = (pred_to_target + target_to_pred) / 2.0
    
    return dice, assd

def get_boundary(binary_mask):
    """
    Extract boundary of a binary mask
    """
    eroded = F.max_pool3d(binary_mask, kernel_size=3, stride=1, padding=1) * binary_mask
    boundary = binary_mask - eroded
    return boundary

def compute_per_channel_dice(pred, target, epsilon=1e-5):
    """
    Compute Dice score for each channel separately
    """
    assert pred.size() == target.size(), "Prediction and target must have the same size"
    
    # Flatten predictions and targets
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    
    # Compute intersection and union
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)
    
    # Compute Dice score
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

def DiceLong(pred, target, num_classes):
    """
    Compute Dice score for long format segmentation
    """
    dice_scores = []
    
    for c in range(1, num_classes):  # Skip background class 0
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice_scores.append((2. * intersection) / union)
        else:
            dice_scores.append(torch.tensor(1.0, device=pred.device))
    
    return torch.stack(dice_scores).mean()

def DiceLong_seg(pred, target, num_classes):
    """
    Compute Dice score for each class in long format segmentation
    """
    dice_scores = []
    
    for c in range(1, num_classes):  # Skip background class 0
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice_scores.append((2. * intersection) / union)
        else:
            dice_scores.append(torch.tensor(1.0, device=pred.device))
    
    return dice_scores

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        
    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target, 
                                      kernel_mul=self.kernel_mul,
                                      kernel_num=self.kernel_num)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss

class MulticlassDiceLossVectorize(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(MulticlassDiceLossVectorize, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, input, target):
        """
        input: (B, C, H, W, D) tensor of predicted probabilities
        target: (B, H, W, D) tensor of ground truth labels (long)
        """
        if input.dim() > 2:
            # (B, C, H, W, D) -> (B, C, H*W*D)
            input = input.view(input.size(0), input.size(1), -1)
            
        if target.dim() == 5:
            # (B, C, H, W, D) -> (B, C, H*W*D)
            target = target.view(target.size(0), target.size(1), -1)
        elif target.dim() == 4:
            # Convert (B, H, W, D) to one-hot encoding
            n_classes = input.size(1)
            target_one_hot = torch.zeros_like(input)
            
            for b in range(target.size(0)):
                for c in range(n_classes):
                    if self.ignore_index is not None and c == self.ignore_index:
                        continue
                    target_one_hot[b, c] = (target[b] == c).float().view(-1)
            
            target = target_one_hot
        
        # Compute Dice coefficient for each class
        intersect = (input * target).sum(dim=2)
        denominator = input.sum(dim=2) + target.sum(dim=2)
        
        # Compute Dice loss
        dice_score = (2. * intersect + 1e-5) / (denominator + 1e-5)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(input.device)
            dice_score = dice_score * weight
        
        # Average over classes and batches
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss

def comput_fig_mask(img_tensor):
    """
    Compute a figure mask for visualization
    """
    # Normalize to [0, 1]
    img = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-5)
    
    # Convert to numpy for visualization
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    return img

class DC_and_CE_loss(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1):
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()
        self.dc = MulticlassDiceLossVectorize()
        
    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

def calculate_intersection(mask1, mask2):
    """
    Calculate intersection between two binary masks
    """
    return (mask1 * mask2).sum().item()

def calculate_union(mask1, mask2):
    """
    Calculate union between two binary masks
    """
    return (mask1 + mask2).clamp(0, 1).sum().item()

def merge_segmentations(seg_list, weights=None):
    """
    Merge multiple segmentations with optional weights
    """
    if weights is None:
        weights = [1.0] * len(seg_list)
    
    merged = torch.zeros_like(seg_list[0])
    for seg, weight in zip(seg_list, weights):
        merged += seg * weight
    
    # Get the class with highest weighted vote
    merged = torch.argmax(merged, dim=1, keepdim=True)
    
    return merged

def consistency_loss(pred1, pred2, weight=1.0):
    """
    Compute consistency loss between two predictions
    """
    return F.mse_loss(pred1, pred2) * weight

def read_csv_to_list(csv_path):
    """
    Read CSV file to list
    """
    import csv
    result = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            result.append(row)
    return result

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

def adjust_learning_rate_power(optimizer, epoch, max_epochs, init_lr, power=0.9):
    """
    Adjust learning rate with polynomial decay
    """
    lr = round(init_lr * np.power(1 - (epoch) / max_epochs, power), 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def checkpointdice(model, optimizer, dice, epoch, save_path):
    """
    Save model checkpoint with dice score
    """
    state = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    filename = os.path.join(save_path, f'best_Dice{dice}.pth')
    torch.save(state, filename)
    print(f'Saved checkpoint at epoch {epoch} with Dice {dice}')