import os
import cudf as pd
from sklearn.model_selection import train_test_split
import Memory_optimize as memory
def Create_directory():
    path='/home/noam/dataSet'
    if not (os.path.exists(path)):
        print("Making directory...")
        os.mkdir(path)
        os.mkdir(f'{path}/Train')
        os.mkdir(f'{path}/Validation')
        os.mkdir(f'{path}/Test')
    return path;

def Split_Train(train_path):
    train_path=os.path.expanduser(train_path)
    path=pd.read_csv(train_path)

    x = path.drop('label', axis=1) 
    y = path['label']
    del path;
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3,shuffle=True, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33,shuffle=True, random_state=42)
    to_pandas = lambda pd: pd.to_pandas()
    x_train, y_train = to_pandas(x_train), to_pandas(y_train)
    x_val,y_val=to_pandas(x_val), to_pandas(y_val)
    x_test,y_test=to_pandas(x_test), to_pandas(y_test)
    memory
    if len(f'{Create_directory()}/Train/')==0:
        x_train.to_csv(f'{Create_directory()}/Train/x_train',index=False)
        y_train.to_csv(f'{Create_directory()}/Train/y_train',index=False)
    
    if len(f'{Create_directory()}/Validation/')==0:
        x_val.to_csv(f'{Create_directory()}/Validation/x_val',index=False)
        y_val.to_csv(f'{Create_directory()}/Validation/y_val',index=False)
    if len(f'{Create_directory()}/Test/')==0:
        x_test.to_csv(f'{Create_directory()}/Test/x_test',index=False)
        y_test.to_csv(f'{Create_directory()}/Test/y_test',index=False)
    
    
    
    
    

