import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from   torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from pathlib import Path
from torch.utils.data import DataLoader

from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    csv_train   = 'my_annotations.csv'
    csv_classes = 'my_class_list.csv'
    csv_val     = 'my_annotations.csv'
    model_path  = 'model_weights_test.pt'
 
    depth       = 18
    batch_size  = 1
    epochs      = 2
    
    evaluate_validation_set = False
    
    # Training dataset
    dataset_train = CSVDataset(train_file=csv_train,
                               class_list=csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    # Validation dataset
    if csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=csv_val,
                                 class_list=csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    # History
    train_hist = []
    valid_hist = []
    eval_hist  = []
    train_hist_header = 'epoch,iteration,loss,classification loss,regression loss,running loss\n'
    valid_hist_header = 'epoch,loss,classification loss,regression loss\n'
    eval_hist_header  = 'epoch,train map,valid map,train iou,valid iou\n'
    train_hist.append(train_hist_header)
    valid_hist.append(valid_hist_header)
    eval_hist.append(eval_hist_header)
    
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss          = []
        validation_loss_tot = []
        validation_loss_cls = []
        validation_loss_reg = []

        # Training
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if iter_num%1000==0:
                    print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                           epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                # Construct history string
                train_info = str(epoch_num) + ',' + str(iter_num) + ',' + str(float(loss)) + ',' + str(float(classification_loss)) + ',' + str(float(regression_loss)) + ',' + str(np.mean(loss_hist)) + '\n'
                train_hist.append(train_info)
                    
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        # Validation
        if csv_val is not None:
            for data_val in dataloader_val:
                try:
                    if torch.cuda.is_available():
                        classification_loss_val, regression_loss_val = retinanet([data_val['img'].cuda().float(), data_val['annot']])
                    else:
                        classification_loss_val, regression_loss_val = retinanet([data_val['img'].float(), data_val['annot']])

                    classification_loss_val = classification_loss_val.mean()
                    regression_loss_val     = regression_loss_val.mean()
                    loss_val                = classification_loss_val + regression_loss_val

                    if bool(loss_val == 0):
                        continue

                    validation_loss_tot.append(float(loss_val))
                    validation_loss_cls.append(float(classification_loss_val))
                    validation_loss_reg.append(float(regression_loss_val))
                    
                    del classification_loss_val
                    del regression_loss_val
                    del loss_val
                except Exception as e:
                    print(e)
                    continue

            print('Epoch: {} | Validation loss: {:1.5f} | Validation Classification loss: {:1.5f} | Validation Regression loss: {:1.5f}'.format(
                   epoch_num, np.mean(validation_loss_tot), np.mean(validation_loss_cls), np.mean(validation_loss_reg)))
        
            # Construct history string
            valid_info = str(epoch_num) + ',' + str(np.mean(validation_loss_tot)) + ',' + str(np.mean(validation_loss_cls)) + ',' + str(np.mean(validation_loss_reg)) + '\n'
            valid_hist.append(valid_info)
        
        # Evaluate validation set
        if csv_val is not None and evaluate_validation_set:

            #print('Evaluating training set')
            #train_mAP_epoch, train_IoU_epoch = csv_eval.evaluate(dataset_train, retinanet, verbose=True)
            
            print('Evaluating validation set')
            valid_mAP_epoch, valid_IoU_epoch = csv_eval.evaluate(dataset_val, retinanet, verbose=True)
            
            # Construct history string
            eval_info = str(epoch_num) + ',' + '0' + ',' + str(valid_mAP_epoch[0][0]) + ',' + '0' + ',' + str(valid_IoU_epoch[0]) + '\n'
            eval_hist.append(eval_info)
            
        scheduler.step(np.mean(epoch_loss))

        #torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, model_path)
    
    # Save training and validation history
    save_history = True
    if save_history:
        
        history_path = Path('results/history')
        if not history_path.parent.is_dir():
            history_path.parent.mkdir(parents=True)
        
        # Save training history
        with open("results/history/training_history.csv", "w") as csv_file:
            for info in train_hist:
                csv_file.write(info)

        # Save validation history
        with open("results/history/validation_history.csv", "w") as csv_file:
            for info in valid_hist:
                csv_file.write(info)

if __name__ == '__main__':
    main()