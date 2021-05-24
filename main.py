import argparse
import logging
# import os
from training import training
from argparse import RawTextHelpFormatter
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description = 'Drinking waste image classification', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-m', '--mode', type = str, help = 
                        '''Choose the program mode: train, default_test, own_test \n\ntrain: you can train the last network layer on train dataset and validate \nthe results on test dataset (STRONGLY NOT RECOMMENDED IF YOU HAVE ONLY CPU) \n\ndefault_test: you can obtain the results of best network model prediction \non arranged test dataset \n\nown_test: you can load your own images in example folder and recieve the prediction of \nwaste type for them ''',
                        default = 'default_test', choices = ['train', 'own_test', 'default_test'], required = False)
    parser.add_argument('-lr', '--learning_rate', type = float, help = 'If you\'ve chosen "train" mode, you can peak a learning rate',
                        default = 0.01, required = False)
    parser.add_argument('-n', '--num_epochs', type = int, help = 'If you\'ve chosen "train" mode, you can peak a number epochs',
                        default = 10, required = False)
    parser.add_argument('-v', '--verbosity', type = int, choices = range(2),
                        default = 0, help = 'Output verbosity', required = False)
    args = parser.parse_args()
    
    levels = [logging.INFO, logging.DEBUG]
    logging.basicConfig(format = u'%(levelname)-8s [%(asctime)s] %(message)s', level = levels[args.verbosity])
    
    logging.debug("Start of program")
    training(args.mode, args.learning_rate, args.num_epochs)
        
if __name__ == "__main__":
    main()    
    