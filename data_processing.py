from pathlib import Path
from zipfile36 import ZipFile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
path = str(Path(__file__).resolve().parent)
os.chdir(path)

def unzip(folder):
    if not os.path.exists('./dataset/' + folder):
        _zip = ZipFile('./dataset/' + folder + '.zip','r')
        _zip.extractall('./dataset/')
        _zip.close()

def create_dataframe(paths, state):
    df = pd.DataFrame(paths)
    df.columns = ['path']
    if not state=='own_test':    
        df['alucan'] = df['path'].apply(lambda x: (x.find('AluCan') >= 0)*1 )
        df['glass'] = df['path'].apply(lambda x: (x.find('Glass') >= 0)*1 )
        df['hdpe'] = df['path'].apply(lambda x: (x.find('HDPEM') >= 0)*1 )
        df['pet'] = df['path'].apply(lambda x: (x.find('PET') >= 0)*1 )
    return df 
   
def preprocessing(state):
    
    if not state == 'own_test':
        unzip(folder = 'test')
        test_images_paths = [os.path.join("./dataset/test/", i) 
                        for i in os.listdir("./dataset/test/")]
        test = create_dataframe(test_images_paths, state)
    
        if state == 'default_test':
            return test
        
        elif state == 'train':
            unzip(folder = 'train')
            train_images_paths = [os.path.join("./dataset/train/", i) 
                                for i in os.listdir("./dataset/train/")]
            train = create_dataframe(train_images_paths, state)
            train = train.sample(frac = 1).reset_index(drop=True)
            
            #делаем доп сплит для валидации на каждой эпохе
            train, val, _,_  = train_test_split(train, train, test_size = 0.1)
            
            train = train.reset_index(drop = True)
            val = val.reset_index(drop = True)

            return test, train, val
        
    else:
        random_test_paths = [os.path.join("./example/", i) 
                        for i in os.listdir("./example/")]
        random_test = create_dataframe(random_test_paths, state)
        return random_test