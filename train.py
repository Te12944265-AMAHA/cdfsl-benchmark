import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, DTD_few_shot, Chest_few_shot, ISIC_few_shot, CropDisease_few_show, EuroSAT_few_shot


# TODO: check when to or not to use val_loader; check argv of each fn
def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, writer):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
       raise ValueError('Unknown optimization, please define by yourself')     

    max_acc = 0

    for epoch in range(start_epoch,stop_epoch):
        model.train() # sets the training mode
        if params.adversarial or params.adaptFinetune:
            model.train_loop_PRODA(epoch, base_loader, optimizer, writer, params=params)
        else:
            model.train_loop(epoch, base_loader, optimizer, writer, params=params) 

        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        # validation
        acc = model.test_loop(val_loader, writer, epoch, params=params)
        # for baseline and baseline++, we don't use validation in default 
        # and we let acc = -1, but we allow options to validate with DB index
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'Adam'

    if params.method in ['baseline'] :

        if params.dataset == "miniImageNet":
        
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader = datamgr.get_data_loader(aug = params.train_aug )

        elif params.dataset == "CUB":
            base_file = configs.data_dir['CUB'] + 'base.json' 
            base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
       
        elif params.dataset == "cifar100":
            base_datamgr    = cifar_few_shot.SimpleDataManager("CIFAR100", image_size, batch_size = 16)
            base_loader    = base_datamgr.get_data_loader( "base" , aug = True )
                
            params.num_classes = 100

        elif params.dataset == 'caltech256':
            base_datamgr  = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader = base_datamgr.get_data_loader(aug = False )
            params.num_classes = 257

        elif params.dataset == "DTD":
            base_datamgr    = DTD_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( aug = True )

        else:
           raise ValueError('Unknown dataset')

        model           = BaselineTrain( model_dict[params.model], params.num_classes)

    elif params.method in ['protonet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        #if params.dataset == "miniImageNet":
        base_datamgr       = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader        = base_datamgr.get_data_loader(aug = params.train_aug)

        # use unlabeled data from these novel domains for adversarial domain adaptation
        # TODO: since the data is unlabeled, we need to modify data manager / data loader
        if params.dataset == "ChestX":
            target_datamgr     = Chest_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)

        elif params.dataset == "EuroSAT":
            target_datamgr     = EuroSAT_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        
        elif params.dataset == "ISIC2018":
            target_datamgr     = ISIC_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        
        elif params.dataset == "CropDiseases":
            target_datamgr     = CropDisease_few_show.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            
        else:
           raise ValueError('Unknown dataset')

        target_loader        = target_datamgr.get_data_loader(novel_file, aug = params.train_aug)

        if params.adversarial or params.adaptFinetune:
            # TODO: check argv
            target_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            target_loader = target_datamgr.get_data_loader(novel_file , aug=True)
            base_loader = [base_loader, target_loader]

        if params.method == 'protonet':
            # TODO: check argv
            if params.adversarial:
                model = ProtoNet(model_dict[params.model], params.test_n_way, params.n_shot, discriminator =
                backbone.Disc_model(params.train_n_way), cosine=params.cosine)
            elif params.adaptFinetune:
                assert (params.adversarial==False)
                model = ProtoNet(model_dict[params.model], params.test_n_way, params.n_shot, adaptive_classifier =
                backbone.Adaptive_Classifier(params.train_n_way), cosine=params.cosine)
            else:
                model = ProtoNet(model_dict[params.model], params.test_n_way, params.n_shot, cosine=params.cosine)
            # original code provided by the benchmark
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)

        # pre_train or warm start
        if params.load_modelpth:
            if 'pretrained-imagenet' in params.load_modelpth:
                model = load_pretrImagenet(model, params.load_modelpth)
            elif ('baseline++' in params.load_modelpth or 'ImgNetPretr' in params.load_modelpth):
                model = load_baselinePP(model, params.load_modelpth)
            else:
                model = load_model(model, params.load_modelpth)
            print('preloading: ', params.load_modelpth)
       
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    save_dir =  configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    logdir = params.checkpoint_dir.replace('checkpoints', 'logs')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model_state_dict = model.state_dict()
            pretr_dict = {k: v for k, v in tmp.items() if k in model_state_dict}
            model_state_dict.update(pretr_dict)
            model.load_state_dict(model_state_dict)

    elif params.warmup: # warmup from pretrained baseline feature, never used in closer look cdfsl paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    # TODO: check argv
    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)