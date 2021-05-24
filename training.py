from data_processing import preprocessing
from dataset import TrainDataset, get_train_transform
from torch.utils.data import DataLoader
import torchvision.models as models
from network import NN
import torch
from trainer import Trainer
import numpy as np
from dataset import TrainDataset, get_test_transform
from torch.utils.data import DataLoader
import logging


def training(state, lr = 0.01, num_epochs = 10):
    
    resnet = models.resnet101(pretrained = True)
    
    for param in resnet.parameters():
        param.requires_grad=False
    
    logging.debug('Weights are frozen')
    our_resnet_model = NN(resnet)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    logging.info(f'The availible device is: {device}')
    our_resnet_model = our_resnet_model.to(device)
    logging.debug(f'Sent pretrained model to {device}')
    test_transform = get_test_transform()
    
    if state == 'train':
        
        test, train, val = preprocessing(state = 'train')
        logging.info(f'Preprocessing of train dataset is done: \n {train.head()}')
        logging.info(f'Preprocessing of test dataset is done: \n {test.head()}')
        logging.debug(f'Preprocessing is done: \n train is {train.shape[0]} \n val is {val.shape[0]} \n test is {test.shape[0]}')
        
        train_transform = get_train_transform()
        
        train_dataset = TrainDataset(train, train_transform)
        val_dataset = TrainDataset(val, train_transform)
        
        train_dataloader =   DataLoader(train_dataset, batch_size = 16, shuffle = True)
        val_dataloader =   DataLoader(val_dataset, batch_size = 16, shuffle = False)
        
        trainer = Trainer( model= our_resnet_model, device = device, lr = lr, ready = False)
        trainer.fit(train_dataloader, val_dataloader, num_epochs = num_epochs)
        
        test_dataset = TrainDataset(test['path'].to_frame(), test_transform, is_test = True)
        test_dataloader =  DataLoader(test_dataset, batch_size = 16, shuffle = False)
    
    else:
        test = preprocessing(state)
        logging.info(f'Preprocessing of test dataset is done: \n {test.head()}')
        test_dataset = TrainDataset(test['path'].to_frame(), test_transform, is_test = True)
        test_dataloader =  DataLoader(test_dataset, batch_size = 16, shuffle = False)
        trainer = Trainer(model = our_resnet_model, device = device, lr = lr, ready = True)
        
       
    test_predictions= trainer.predict(test_dataloader)
    predictions = np.around(test_predictions)
    
    if state == 'own_test':
        variants = ['alucan', 'glass', 'hdpe', 'pet']
        test['answer'] = [variants[np.where(prediction == 1)[0][0]] for prediction in predictions]
        logging.info(f'Prediction for test dataset: \n {test}')
        test.to_csv('./results/results.tsv', sep = '\t')
    else:
        count = 0
        for i, row in test[['alucan', 'glass', 'hdpe', 'pet']].iterrows():
            if (predictions[i] == row.values).all():
                count += 1
        logging.info(f'Accuracy: {count/test.shape[0]}')  
