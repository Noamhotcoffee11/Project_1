# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def main():
    import Memory_optimize
    import Data_Manipolation as dp
    Memory_optimize
    dp.Create_directory()
    path='~/train_features.csv'
    dp.Split_Train(path)


    
if __name__=='__main__':
    main()
