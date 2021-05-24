from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import os
from glob import glob
import logging

class Trainer:
    def __init__(self, model, device, lr, ready = False):
        self.model = model
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam( [param for param in self.model.parameters() if param.requires_grad],
                                    lr = lr)
        if ready:
            checkpoint = torch.load('./results/best_result.pt', map_location = device)
            self.model.load_state_dict(checkpoint['state_dict'])
        
    def fit(self, train_dataloader, val_dataloader, num_epochs):
        logging.info('The training is started')
        total = 0
        correct= 0
        
        loss_values = []
        accuracy_values = []
        
        valid_results = []
        
        for epoch in range(num_epochs):
            self.model.train()
            batch_number = 0
            # loss_values_batch = []
            for x, y in train_dataloader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                l = self.loss(outputs, torch.max(y, 1)[1].long()) 
                l.backward()
                self.optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == torch.max(y, 1)[1]).sum().item()
                loss_value = l.item()
                loss_values.append(loss_value)
                accuracy_values.append(correct/total)

               
                if batch_number%10 ==0:
                    logging.debug(f"batch number {batch_number}, loss_value: {loss_value}")
                    current_accuracy = correct/total
                    logging.debug(f"current_accuracy: {current_accuracy}")                   
                batch_number+= 1
                
            epoch_accuracy = correct/total
            logging.debug(f"epoch_accuracy: {epoch_accuracy}")
            logging.info(f"end of epoch {epoch}")
            
            epoch_val_loss = []
            
            correct = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                for x, y in val_dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs = self.model(x)
                    l = self.loss(outputs, torch.max(y, 1)[1].long())
                    loss_value = l.item()
                    epoch_val_loss.append(loss_value)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == torch.max(y, 1)[1]).sum().item()
                    
            logging.info(f"Total {total} Correct {correct} Accuracy {correct/total}")
            valid_results.append(correct/total)
            #save the model
            checkpoint = {
            'epoch': epoch,
            'valid_loss_min': correct/total,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
            
            torch.save(checkpoint, './results/current_checkpoint_' + str(epoch) + '.pt')
            
        logging.getLogger('matplotlib.font_manager').disabled = True    
        if num_epochs > 1:
            plt.figure(figsize=(5,5))
            plt.plot([i+1 for i in range(len(valid_results))], valid_results)
            # plt.show()
            plt.savefig('./results/training.png')
            
        _index = valid_results.index(max(valid_results))
        path = './results/current_checkpoint_' + str(_index) + '.pt'
        if os.path.exists('./results/best_result.pt'):
            os.remove('./results/best_result.pt')
        os.rename(path, './results/best_result.pt')
        
        for u in glob('./results/current_checkpoint_*'):
            os.remove(u)
        logging.info('The training is over')    
        
    def predict(self, test_dataloader):
        logging.info('The prediction is started')
        self.model.eval()
        predictions = torch.tensor([]) 
        with torch.no_grad():
            for x in test_dataloader:
                x = x.to(self.device)
                outputs = torch.nn.functional.softmax(self.model(x))
                predictions = torch.cat([predictions,outputs.detach().cpu()])
        return predictions.numpy()