import torch
import torch.nn.functional as F
import math
import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from numpy.linalg import eig
from medpy.io import load, save, header
import math
import glob
from scipy.ndimage import morphology
import torch.nn.functional as nnf
import random
import SimpleITK as sitk
import itk
import shutil


class KL:
    """
    Kullback–Leibler divergence for probabilistic flows.
    """

    def __init__(self, prior_lambda, flow_vol_shape):
        self.prior_lambda = prior_lambda
        self.flow_vol_shape = flow_vol_shape # [B C H W D]
        self.D = None

    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewhere.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """
        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1  #3x3x3 filter

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature

        filt = np.zeros([ndims, ndims] + [3] * ndims )
        for i in range(ndims):
            filt[i, i, ...] = filt_inner

        return filt #shape=3,3,(3,3,3)

    def _degree_matrix(self, vol_shape):  #compute the degree matrix
        # get shape stats
        ndims = len(vol_shape)  #ndims=3
        sz = [ndims, *vol_shape] #[3,160,192,224]

        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # prepare tf filter
        z = torch.ones([1] + sz) #ones([1,3,160,192,224])
        filt_torch = torch.Tensor(self._adj_filt(ndims)) #shape=3,3,(3,3,3)

        return conv_fn(z, filt_torch, stride=1, padding=1)

    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = list(y_pred.shape)[2:] #[160,192,224]
        ndims = len(vol_shape) #ndims=3

        sm = 0
        for i in range(ndims):
            d = i + 2 #  2 3 4
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y_pred.permute(r)  #y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += torch.mean(df * df)

        return 0.5 * sm / ndims

    def loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs

        ndims = len(y_pred.shape) - 2  #ndims=3
        mean = y_pred[:, 0:ndims,...] #channel is the second axis in pytorch
        log_sigma = y_pred[:, ndims:,...]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data
        if self.D is None:
            #self.D = self._degree_matrix(self.flow_vol_shape).cuda(3) #flow_vol_shape=[160,192,224], D.shape=?
            self.D = self._degree_matrix(self.flow_vol_shape).cuda(1)
        # sigma terms
        sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma
        sigma_term = torch.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        # ndims because we averaged over dimensions as well
        return 0.5 * ndims * (sigma_term + prec_term)


class MSE_sigma:
    ### 2500 times of the original MSE Loss
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, image_sigma=1.0):
        self.image_sigma = image_sigma

    def loss(self, y_true, y_pred):
        return 1.0 / (self.image_sigma**2) * torch.mean((y_true - y_pred) ** 2)





def comput_fig_cas(warpimg, fiximg, movingimg, vxm_mask, affineimg, affinemask, infixed_mask, inmoving_mask):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    affinemask = affinemask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    affineimg = affineimg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    vxm_mask = vxm_mask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    img = np.zeros((24, infixed_mask.shape[0], infixed_mask.shape[1]))
    img[[0, 8, 16], :, :] = fiximg.transpose(2, 0, 1)
    img[[1, 9, 17], :, :] = affineimg.transpose(2, 0, 1)
    img[[2, 10, 18], :, :] = warpimg.transpose(2, 0, 1)
    img[[3, 11, 19], :, :] = movingimg.transpose(2, 0, 1)
    img[[4, 12, 20], :, :] = infixed_mask.transpose(2, 0, 1)
    img[[5, 13, 21], :, :] = affinemask.transpose(2, 0, 1)
    img[[6, 14, 22], :, :] = vxm_mask.transpose(2, 0, 1)
    img[[7, 15, 23], :, :] = inmoving_mask.transpose(2, 0, 1)

    fig = plt.figure(figsize=(10, 18.75), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(24):
        if i in [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]:
            plt.subplot(6, 4, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg == 0] = [0, 0, 0]
            img2show[showimg == 1] = [255, 0, 0]

            plt.imshow(img2show)

        else:
            plt.subplot(6, 4, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
    return fig


def comput_fig_casnoaf(warpimg, fiximg, movingimg, vxm_mask, infixed_mask, inmoving_mask):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 64:67]
    vxm_mask = vxm_mask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 64:67]
    img = np.zeros((18, infixed_mask.shape[0], infixed_mask.shape[1]))
    img[[0, 6, 12], :, :] = fiximg.transpose(2, 0, 1)
    img[[1, 7, 13], :, :] = warpimg.transpose(2, 0, 1)
    img[[2, 8, 14], :, :] = movingimg.transpose(2, 0, 1)
    img[[3, 9, 15], :, :] = infixed_mask.transpose(2, 0, 1)
    img[[4, 10, 16], :, :] = vxm_mask.transpose(2, 0, 1)
    img[[5, 11, 17], :, :] = inmoving_mask.transpose(2, 0, 1)

    fig = plt.figure(figsize=(7.5, 18.75), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(18):
        if i in [3, 4, 5, 9, 10, 11, 15, 16, 17]:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg == 0] = [0, 0, 0]
            img2show[showimg == 1] = [255, 0, 0]

            plt.imshow(img2show)

        else:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
    return fig

def comput_fig_liver_mask(warpimg, fiximg, movingimg,vxm_mask, infixed_mask,inmoving_mask):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 45:48]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 45:48]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 45:48]
    vxm_mask = vxm_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    img = np.zeros((18,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,6,12],:, :] = fiximg.transpose(2,0,1)
    img[[1,7,13],:, :] = warpimg.transpose(2,0,1)
    img[[2,8,14],:, :] = movingimg.transpose(2,0,1)
    img[[3,9,15],:, :] = infixed_mask.transpose(2,0,1)
    img[[4,10,16],:, :] = vxm_mask.transpose(2,0,1)
    img[[5,11,17],:, :] = inmoving_mask.transpose(2,0,1)
    
    fig = plt.figure(figsize=(7.5,18.75), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(18):
        if i in [3,4,5,9,10,11,15,16,17]:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg==0] = [0,0,0]
            img2show[showimg==1] = [255,0,0]
            plt.imshow(img2show)
        else:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
    return fig



def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented

def resample(sitk_image, out_spacing=(1, 1, 1), is_label=False):
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample_worker = sitk.ResampleImageFilter()
    resample_worker.SetDefaultPixelValue(-1024)
    resample_worker.SetOutputOrigin(sitk_image.GetOrigin())
    resample_worker.SetOutputDirection(sitk_image.GetDirection())
    resample_worker.SetSize(out_size)
    resample_worker.SetOutputSpacing(out_spacing)
    if is_label:
        resample_worker.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_worker.SetInterpolator(sitk.sitkBSpline)
    resampled = resample_worker.Execute(sitk_image)
    return resampled

def adust_learningrate_design(optimizer,epoch,lr,n = 10):
    if epoch <= 20:
        lr = lr*(0.5**(epoch//n))
    elif epoch <= 100:
        lr = lr*(0.5**(epoch//n))
    else:
        lr = 0.000001
    for param in optimizer.param_groups:
        param['lr'] = lr
    return lr


def adjust_learning_rate_power(optimizer, epoch, max_epochs, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power( 1 - (epoch) / max_epochs, power), 8)


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        # y_pred[atlas_dil == 0] = 0
        J = y_pred
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda:0")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class NCC_mask:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Io = y_true
        Jo = y_pred
        bound = torch.nonzero(y_true,as_tuple=True)
        x =bound[0]
        y =bound[1]
        z = bound[2]
        xl = torch.min(x)
        xu = torch.max(x)
        yl = torch.min(y)
        yu = torch.max(y)
        zl = torch.min(z)
        zu = torch.max(z)
        I = Io[xl-10:xu+10, yl - 10: yu +10, zl: zu+10]
        J = Jo[xl-10:xu+10, yl - 10: yu +10, zl: zu+10]
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda:0")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)




# class MSE:
#     """
#     Mean squared error loss.
#     """
#     def loss(self, y_true, y_pred):
#         b = torch.sum(torch.logical_not(torch.logical_and((y_true == 0), (y_pred == 0))))
#         # return torch.mean((y_true - y_pred) ** 2)
#         return torch.sum((y_true - y_pred) ** 2) / b

class MSE_nomask:
    """
    Mean squared error loss.
    """
    def loss(self, y_true, y_pred):
        # b = torch.sum(torch.logical_not(torch.logical_and((y_true == 0), (y_pred == 0))))
        return torch.mean((y_true - y_pred) ** 2)
        # return torch.sum((y_true - y_pred) ** 2) / b

# class MSE_mask:
#     """
#     Mean squared error loss.
#     """
#     def loss(self, y_true, y_pred, loss_mask_binary, erosion_mask):
#         loss_mask = loss_mask_binary * 10
#         loss_mask[loss_mask == 0] = 1
#         loss_mask[erosion_mask == 1] = 5
#         loss_mse_pixel = (y_true - y_pred) ** 2
#         loss_mse_mask = loss_mse_pixel * loss_mask
#         return torch.sum(loss_mse_mask) / torch.sum(loss_mask_binary + loss_mask_binary)





# class MSE_mask:
#     """
#     Mean squared error loss.
#     """
#     def loss(self, y_true, y_pred, y_true_mask, y_pred_mask):
#         maskf = torch.zeros_like(y_true_mask)
#         maskf[y_true_mask == 1] = 1
#         maskf.requires_grad = False
#         maskm = torch.zeros_like(y_pred_mask)
#         maskm[y_pred_mask == 1] = 1
#         maskm.requires_grad = False
#         y_true = y_true * maskf
#         y_pred = y_pred * maskm
#         return torch.mean((y_true - y_pred) ** 2)




class L1:
    """
    Mean squared error loss.
    """
    def loss(self, y_true, y_pred):
        
        return torch.mean((y_true - y_pred) ** 2)
        # return torch.sum(abs(y_true - y_pred)) / b

class Dice:
    
    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

def comput_fig(warpimg, fiximg, movingimg):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 125:130]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 125:130]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 125:130]
    img = np.concatenate((fiximg, movingimg,warpimg), 2)
    fig = plt.figure(figsize=(15,12), dpi=50)
    plt.subplots_adjust(wspace=0.0001, hspace=0.01)
    for i in range(img.shape[2]):
        plt.subplot(3, 5, i + 1)
        plt.axis('off')
        plt.imshow(img[:, :, i].T, cmap='gray')
    return fig

def comput_fig(warpimg, fiximg, movingimg):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 125:130]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 125:130]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 125:130]
    img = np.concatenate((fiximg, movingimg,warpimg), 2)
    fig = plt.figure(figsize=(15,12), dpi=50)
    plt.subplots_adjust(wspace=0.0001, hspace=0.01)
    for i in range(img.shape[2]):
        plt.subplot(3, 5, i + 1)
        plt.axis('off')
        plt.imshow(img[:, :, i].T, cmap='gray')
    return fig



def comput_fig_liver(warpimg_gray, warpimg_mask, fiximg, movingimg_mask, movingimg_gray):
    warpimg_mask = warpimg_mask.detach().cpu().numpy()[0, 0, :,80:85, :]
    warpimg_gray = warpimg_gray.detach().cpu().numpy()[0, 0, :,80:85, :]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :,80:85, :]
    movingimg_mask = movingimg_mask.detach().cpu().numpy()[0, 0, :, 80:85, :]
    movingimg_gray = movingimg_gray.detach().cpu().numpy()[0, 0, :, 80:85, :]
    img = np.concatenate((fiximg, movingimg_mask,movingimg_gray, warpimg_mask, warpimg_gray), 1)
    fig = plt.figure(figsize=(20.5,27), dpi=50)
    plt.subplots_adjust(wspace=0.0001, hspace=0.001)
    for i in range(img.shape[1]):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    return fig

def comput_fig_mask(fiximg):
    
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :,80:85, :]
    img = np.concatenate((fiximg, fiximg, fiximg, fiximg, fiximg), 1)
    fig = plt.figure(figsize=(20.5,27), dpi=50)
    plt.subplots_adjust(wspace=0.0001, hspace=0.001)
    for i in range(img.shape[1]):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        showimg = img[:, i, :]
        img2show = np.zeros((img[:, i, :].shape[0], img[:, i, :].shape[1], 3))
        img2show[showimg==0] = [0,0,0]
        img2show[showimg==1] = [255,0,0] #red hip
        img2show[showimg==2] = [0,255,0] #green
        img2show[showimg==3] = [255,0,255] # yellow
        # img2show[showimg==3] = [255,255,0] # yellow
        img2show[showimg==4] = [0,0,255] # blue
        plt.imshow(img2show)
    return fig


def comput_figsegreg(warpimg, fiximg, movingimg,seg_out, warp_seg, 
        warp_moving_mask, inmoving_mask, infixed_mask):
    
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    
    seg_outpng = seg_out.detach().cpu().numpy()[0, 0, :, :, 100:103]
    print(np.unique(seg_outpng))
    warp_seg_png = warp_seg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    print(np.unique(warp_seg_png))
    warp_moving_mask = warp_moving_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    print(np.unique(warp_moving_mask))
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    
    # img =  np.concatenate((fiximg, movingimg, warpimg), 2)
    img = np.zeros((27,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,3,6],:, :] = fiximg.transpose(2,0,1)
    img[[1,4,7],:, :] = infixed_mask.transpose(2,0,1)
    img[[2,5,8],:, :] = infixed_mask.transpose(2,0,1)
    img[[9,12,15],:, :] = movingimg.transpose(2,0,1)
    img[[10,13,16],:, :] = inmoving_mask.transpose(2,0,1)
    img[[11,14,17],:, :] = seg_outpng.transpose(2,0,1)
    img[[18,21,24],:, :] = warpimg.transpose(2,0,1)
    img[[19,22,25],:, :] = warp_moving_mask.transpose(2,0,1)
    img[[20,23,26],:, :] = warp_seg_png.transpose(2,0,1)
    
    fig = plt.figure(figsize=(18.75,7.5), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(27):
        if i in [0,3,6,9,12,15,18,21,24]:
            plt.subplot(3, 9, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
        else:
            plt.subplot(3, 9, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg==0] = [0,0,0]
            img2show[showimg==1] = [255,0,0]
            img2show[showimg==2] = [0,255,0]
            img2show[showimg==3] = [0,0,255]
            img2show[showimg==4] = [255,255,0]
            plt.imshow(img2show)
    return fig

def comput_bireg_oasis(infixed,warped_MF,warped_FM_MF,inmoving,warped_FM, warped_MF_FM, \
            infixed_mask, warp_MF_mask, warp_FM_MF_mask, inmoving_mask, warp_FM_mask, warp_MF_FM_mask):
    
    infixed = infixed.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warped_MF = warped_MF.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warped_FM_MF = warped_FM_MF.detach().cpu().numpy()[0, 0, :, :, 100:103]
    inmoving = inmoving.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warped_FM = warped_FM.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warped_MF_FM = warped_MF_FM.detach().cpu().numpy()[0, 0, :, :, 100:103]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warp_MF_mask = warp_MF_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warp_FM_MF_mask = warp_FM_MF_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warp_FM_mask = warp_FM_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    warp_MF_FM_mask = warp_MF_FM_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    
    # img =  np.concatenate((fiximg, movingimg, warpimg), 2)
    img = np.zeros((36,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,12,24],:, :] = infixed.transpose(2,0,1)
    img[[1,13,25],:, :] = warped_MF.transpose(2,0,1)
    img[[2,14,26],:, :] = warped_FM_MF.transpose(2,0,1)
    img[[3,15,27],:, :] = inmoving.transpose(2,0,1)
    img[[4,16,28],:, :] = warped_FM.transpose(2,0,1)
    img[[5,17,29],:, :] = warped_MF_FM.transpose(2,0,1)
    img[[6,18,30],:, :] = infixed_mask.transpose(2,0,1)
    img[[7,19,31],:, :] = warp_MF_mask.transpose(2,0,1)
    img[[8,20,32],:, :] = warp_FM_MF_mask.transpose(2,0,1)
    img[[9,21,33],:, :] = inmoving_mask.transpose(2,0,1)
    img[[10,22,34],:, :] = warp_FM_mask.transpose(2,0,1)
    img[[11,23,35],:, :] = warp_MF_FM_mask.transpose(2,0,1)
    
    fig = plt.figure(figsize=(15,15), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(36):
        if i in [0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28,29]:
            plt.subplot(6, 6, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
        else:
            plt.subplot(6, 6, i + 1)
            plt.axis('off')

            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3)).astype(np.int)
            print(img2show.dtype)
            img2show[showimg == 0] = [0, 0, 0]
            img2show[showimg == 1] = [245, 245, 245]
            img2show[showimg == 2] = [205, 62, 78]
            img2show[showimg == 3] = [120, 18, 134]
            img2show[showimg == 4] = [196, 58, 250]

            img2show[showimg == 5] = [220, 248, 164]

            img2show[showimg == 6] = [230, 148, 34]
            img2show[showimg == 7] = [0, 118, 14]
            img2show[showimg == 8] = [122, 186, 220]
            img2show[showimg == 9] = [236, 13, 176]

            img2show[showimg == 10] = [12, 48, 255]
            img2show[showimg == 11] = [204, 182, 142]
            img2show[showimg == 12] = [42, 204, 164]
            img2show[showimg == 13] = [119, 159, 176]
            img2show[showimg == 14] = [220, 216, 20]

            img2show[showimg == 15] = [103, 255, 255]

            img2show[showimg == 16] = [255, 165, 0]
            img2show[showimg == 17] = [165, 42, 42]
            img2show[showimg == 18] = [160, 32, 240]
            img2show[showimg == 19] = [0, 200, 200]

            img2show[showimg == 20] = [245, 246, 25]
            img2show[showimg == 21] = [165, 42, 42]
            img2show[showimg == 22] = [196, 58, 250]
            img2show[showimg == 23] = [196, 58, 250]
            img2show[showimg == 24] = [220, 248, 164]

            img2show[showimg == 25] = [230, 148, 34]
            img2show[showimg == 26] = [0, 118, 14]
            img2show[showimg == 27] = [122, 186, 22]
            img2show[showimg == 28] = [236, 13, 176]
            img2show[showimg == 29] = [12, 48, 255]

            img2show[showimg == 30] = [220, 216, 20]
            img2show[showimg == 31] = [103, 55, 255]
            img2show[showimg == 32] = [255, 165, 0]
            img2show[showimg == 33] = [165, 42, 42]
            img2show[showimg == 34] = [160, 32, 240]

            img2show[showimg == 35] = [255, 165, 0]

            plt.imshow(img2show)

    return fig


def comput_bireg_nirep(infixed,warped_MF,warped_FM_MF,inmoving,warped_FM, warped_MF_FM, \
            infixed_mask, warp_MF_mask, warp_FM_MF_mask, inmoving_mask, warp_FM_mask, warp_MF_FM_mask):
    
    infixed = infixed.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warped_MF = warped_MF.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warped_FM_MF = warped_FM_MF.detach().cpu().numpy()[0, 0, :, :, 107:110]
    inmoving = inmoving.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warped_FM = warped_FM.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warped_MF_FM = warped_MF_FM.detach().cpu().numpy()[0, 0, :, :, 107:110]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warp_MF_mask = warp_MF_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warp_FM_MF_mask = warp_FM_MF_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warp_FM_mask = warp_FM_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    warp_MF_FM_mask = warp_MF_FM_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    
    # img =  np.concatenate((fiximg, movingimg, warpimg), 2)
    img = np.zeros((36,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,12,24],:, :] = infixed.transpose(2,0,1)
    img[[1,13,25],:, :] = warped_MF.transpose(2,0,1)
    img[[2,14,26],:, :] = warped_FM_MF.transpose(2,0,1)
    img[[3,15,27],:, :] = inmoving.transpose(2,0,1)
    img[[4,16,28],:, :] = warped_FM.transpose(2,0,1)
    img[[5,17,29],:, :] = warped_MF_FM.transpose(2,0,1)
    img[[6,18,30],:, :] = infixed_mask.transpose(2,0,1)
    img[[7,19,31],:, :] = warp_MF_mask.transpose(2,0,1)
    img[[8,20,32],:, :] = warp_FM_MF_mask.transpose(2,0,1)
    img[[9,21,33],:, :] = inmoving_mask.transpose(2,0,1)
    img[[10,22,34],:, :] = warp_FM_mask.transpose(2,0,1)
    img[[11,23,35],:, :] = warp_MF_FM_mask.transpose(2,0,1)
    
    fig = plt.figure(figsize=(15,15), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(36):
        if i in [0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28,29]:
            plt.subplot(6, 6, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
        else:
            plt.subplot(6, 6, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg==0] = [0,0,0]
            img2show[showimg==15] = [255,0,0]
            img2show[showimg==23] = [0,255,0]
            img2show[showimg==11] = [0,0,255]
            img2show[showimg==33] = [255,255,0]
            plt.imshow(img2show)
    return fig


def comput_fig_nirep(warpimg, fiximg, movingimg,vxm_mask, infixed_mask,inmoving_mask):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 107:110]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 107:110]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 107:110]
    vxm_mask = vxm_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 107:110]
    img = np.zeros((18,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,6,12],:, :] = fiximg.transpose(2,0,1)
    img[[1,7,13],:, :] = warpimg.transpose(2,0,1)
    img[[2,8,14],:, :] = movingimg.transpose(2,0,1)
    img[[3,9,15],:, :] = infixed_mask.transpose(2,0,1)
    img[[4,10,16],:, :] = vxm_mask.transpose(2,0,1)
    img[[5,11,17],:, :] = inmoving_mask.transpose(2,0,1)
    
    fig = plt.figure(figsize=(7.5,18.75), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(18):
        if i in [3,4,5,9,10,11,15,16,17]:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg==0] = [0,0,0]
            img2show[showimg==15] = [255,0,0]
            img2show[showimg==23] = [0,255,0]
            img2show[showimg==11] = [0,0,255]
            img2show[showimg==33] = [255,255,0]
            
            plt.imshow(img2show)
            
        else:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
    return fig

def comput_fig_oasis(warpimg, fiximg, movingimg,vxm_mask, infixed_mask,inmoving_mask):
    warpimg = warpimg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    fiximg = fiximg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    movingimg = movingimg.detach().cpu().numpy()[0, 0, :, :, 100:103]
    vxm_mask = vxm_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 100:103]
    img = np.zeros((18,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,6,12],:, :] = fiximg.transpose(2,0,1)
    img[[1,7,13],:, :] = warpimg.transpose(2,0,1)
    img[[2,8,14],:, :] = movingimg.transpose(2,0,1)
    img[[3,9,15],:, :] = infixed_mask.transpose(2,0,1)
    img[[4,10,16],:, :] = vxm_mask.transpose(2,0,1)
    img[[5,11,17],:, :] = inmoving_mask.transpose(2,0,1)
    
    fig = plt.figure(figsize=(7.5,18.75), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(18):
        if i in [3,4,5,9,10,11,15,16,17]:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3)).astype(np.int)
            print(img2show.dtype)
            img2show[showimg == 0] = [0, 0, 0]
            img2show[showimg == 1] = [245, 245, 245]
            img2show[showimg == 2] = [205, 62, 78]
            img2show[showimg == 3] = [120, 18, 134]
            img2show[showimg == 4] = [196, 58, 250]

            img2show[showimg == 5] = [220, 248, 164]

            img2show[showimg == 6] = [230, 148, 34]
            img2show[showimg == 7] = [0, 118, 14]
            img2show[showimg == 8] = [122, 186, 220]
            img2show[showimg == 9] = [236, 13, 176]

            img2show[showimg == 10] = [12, 48, 255]
            img2show[showimg == 11] = [204, 182, 142]
            img2show[showimg == 12] = [42, 204, 164]
            img2show[showimg == 13] = [119, 159, 176]
            img2show[showimg == 14] = [220, 216, 20]

            img2show[showimg == 15] = [103, 255, 255]

            img2show[showimg == 16] = [255, 165, 0]
            img2show[showimg == 17] = [165, 42, 42]
            img2show[showimg == 18] = [160, 32, 240]
            img2show[showimg == 19] = [0, 200, 200]

            img2show[showimg == 20] = [245, 246, 25]
            img2show[showimg == 21] = [165, 42, 42]
            img2show[showimg == 22] = [196, 58, 250]
            img2show[showimg == 23] = [196, 58, 250]
            img2show[showimg == 24] = [220, 248, 164]

            img2show[showimg == 25] = [230, 148, 34]
            img2show[showimg == 26] = [0, 118, 14]
            img2show[showimg == 27] = [122, 186, 22]
            img2show[showimg == 28] = [236, 13, 176]
            img2show[showimg == 29] = [12, 48, 255]

            img2show[showimg == 30] = [220, 216, 20]
            img2show[showimg == 31] = [103, 55, 255]
            img2show[showimg == 32] = [255, 165, 0]
            img2show[showimg == 33] = [165, 42, 42]
            img2show[showimg == 34] = [160, 32, 240]

            img2show[showimg == 35] = [255, 165, 0]

            plt.imshow(img2show)
            
        else:
            plt.subplot(6, 3, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
    return fig

def comput_bireg_liver(infixed,warped_MF,warped_FM_MF,inmoving,warped_FM, warped_MF_FM, \
            infixed_mask, warp_MF_mask, warp_FM_MF_mask, inmoving_mask, warp_FM_mask, warp_MF_FM_mask):
    
    infixed = infixed.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warped_MF = warped_MF.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warped_FM_MF = warped_FM_MF.detach().cpu().numpy()[0, 0, :, :, 45:48]
    inmoving = inmoving.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warped_FM = warped_FM.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warped_MF_FM = warped_MF_FM.detach().cpu().numpy()[0, 0, :, :, 45:48]
    infixed_mask = infixed_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warp_MF_mask = warp_MF_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warp_FM_MF_mask = warp_FM_MF_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    inmoving_mask = inmoving_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warp_FM_mask = warp_FM_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    warp_MF_FM_mask = warp_MF_FM_mask.detach().cpu().numpy()[0, 0, :, :, 45:48]
    
    # img =  np.concatenate((fiximg, movingimg, warpimg), 2)
    img = np.zeros((36,infixed_mask.shape[0],infixed_mask.shape[1]))
    img[[0,12,24],:, :] = infixed.transpose(2,0,1)
    img[[1,13,25],:, :] = warped_MF.transpose(2,0,1)
    img[[2,14,26],:, :] = warped_FM_MF.transpose(2,0,1)
    img[[3,15,27],:, :] = inmoving.transpose(2,0,1)
    img[[4,16,28],:, :] = warped_FM.transpose(2,0,1)
    img[[5,17,29],:, :] = warped_MF_FM.transpose(2,0,1)
    img[[6,18,30],:, :] = infixed_mask.transpose(2,0,1)
    img[[7,19,31],:, :] = warp_MF_mask.transpose(2,0,1)
    img[[8,20,32],:, :] = warp_FM_MF_mask.transpose(2,0,1)
    img[[9,21,33],:, :] = inmoving_mask.transpose(2,0,1)
    img[[10,22,34],:, :] = warp_FM_mask.transpose(2,0,1)
    img[[11,23,35],:, :] = warp_MF_FM_mask.transpose(2,0,1)
    
    fig = plt.figure(figsize=(15,15), dpi=100)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(36):
        if i in [0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28,29]:
            plt.subplot(6, 6, i + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :].T, cmap='gray')
        else:
            plt.subplot(6, 6, i + 1)
            plt.axis('off')
            showimg = img[i, :, :].T
            img2show = np.zeros((img[i, :, :].shape[1], img[i, :, :].shape[0], 3))
            img2show[showimg==0] = [0,0,0]
            img2show[showimg==1] = [255,0,0]
            # img2show[showimg==2] = [0,255,0]
            # img2show[showimg==3] = [0,0,255]
            # img2show[showimg==4] = [255,255,0]
            plt.imshow(img2show)
    return fig


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2'):
        self.penalty = penalty

    def loss(self, y_true, y_pred):
        bound = torch.nonzero(y_true,as_tuple=True)
        x =bound[0]
        y =bound[1]
        z = bound[2]
        xl = torch.min(x)
        xu = torch.max(x)
        yl = torch.min(y)
        yu = torch.max(y)
        zl = torch.min(z)
        zu = torch.max(z)
        y_pred = y_pred[xl:xu+1, yl: yu +1, zl: zu+1]
        y_true = y_true[xl:xu+1, yl: yu +1, zl: zu+1]
        # dypre = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        # dxpre = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        # dzpre = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        # dytrue = torch.abs(y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :]) 
        # dxtrue = torch.abs(y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :]) 
        # dztrue = torch.abs(y_true[:, :, :, :, 1:] - y_true[:, :, :, :, :-1]) 
        dypre = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
        dxpre = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
        dzpre = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]

        dytrue = y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :]
        dxtrue = y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :] 
        dztrue = y_true[:, :, :, :, 1:] - y_true[:, :, :, :, :-1] 
        
        if self.penalty == 'l2':
            dy = (torch.abs(dypre) - torch.abs(dytrue)) * (torch.abs(dypre) - torch.abs(dytrue))
            dx = (torch.abs(dxpre) - torch.abs(dxtrue)) * (torch.abs(dxpre) - torch.abs(dxtrue))
            dz = (torch.abs(dzpre) - torch.abs(dztrue)) * (torch.abs(dzpre) - torch.abs(dztrue))
            d = torch.mean(dy) + torch.mean(dx) + torch.mean(dz)
        else:
            d = torch.mean(torch.abs(dypre - dytrue)) + torch.mean(torch.abs(dxpre - dxtrue)) + torch.mean(torch.abs(dzpre - dztrue))
        grad = d

        return grad

def checkpointdice(model, optimizer, dice, epoch, name):
    # Save checkpoint.
    print('Saving..')
    state = {
        'model': model.state_dict(),
        'best_dice': dice,
        'epoch': epoch + 1,
        'opt': optimizer.state_dict()
    }
    # # day_hour_minute = time.strftime("%d%H%M", time.localtime())
    # # if not os.path.isdir('checkpoint_%s'%(name)):
    # #     os.mkdir('checkpoint_%s'%(name))
    # # torch.save(state, '%s/E%s_Dice%s.pth' % (name, str(epoch + 1).zfill(3), dice))
    files = glob.glob(name + '/*.pth')
    files.sort(key=lambda x: os.path.getmtime(x))
    if len(files) > 10:
        for f in files[:-2]:
            os.remove(f)
    if check_storage(os.path.dirname(os.path.abspath('%s/best_Dice%s.pth' % (name, dice))), 500):
        torch.save(state, '%s/best_Dice%s.pth' % (name, dice))
    else:
        print("*** Not enough space in the directory to save the model ****")

def check_storage(path, required_space):
    stat = shutil.disk_usage(path)
    # Convert required space to bytes
    required_bytes = required_space * 1024 * 1024 # assuming required_space is in MB
    return stat.free >= required_bytes

##from voxelmorph
def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else: # must be 2 
        
        dfdx = J[0]
        dfdy = J[1] 
        
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]