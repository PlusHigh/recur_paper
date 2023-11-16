import random
import glob
import pickle
from util import load_data

"""
    调用load_original_data函数，load_original_data函数会调用dataset_split函数
    dataset_split函数读取原始数据集，并进行数据划分，划分为训练集、验证集、测试集，比例为6:3:1
"""
from util import load_original_data
load_original_data(test_ratio=0.3,val_ratio=0.1, seed=4325, GP_model=True)


"""
根据type的data生成了data/original_dataset_0.60_0.30_0.10，但这没有进行K折划分，create_dataset函数会数据增强，根据aug_val参数决定是否增强验证集
"""
from util import create_dataset 
for i in range(0,10):   
    create_dataset('data/original_dataset_0.60_0.30_0.10', class_size=1875, down_sample=False, aug_val=True,image=False, save_weight=True, instance=i)



# data_prepare.py 中定义新的函数，用于K折划分
def K_fold_split(K=10, seed=0, fold=0):
    test_ratio = 1/K
    random.seed(seed)
    #不同折seed必须一样
    class_folders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    total_train_names = []
    total_test_names = []
    total_val_names = []
    for i, folder in enumerate(class_folders):
        name_list = glob.glob('data/original_data/type/%s/*.dat'%folder)
        random.shuffle(name_list)
        test_size = int(len(name_list) / K)
        val_size = test_size
        test_names = name_list[fold*test_size:(fold+1)*test_size]
        val_and_train_names = list(set(name_list)-set(test_names))
        val_names = random.sample(val_and_train_names, val_size)
        train_names = list(set(val_and_train_names)-set(val_names))
        total_train_names.append(train_names)
        total_test_names.append(test_names)
        total_val_names.append(val_names)
    return total_train_names, total_val_names, total_test_names

# 调用K_fold_split函数，生成K折划分的数据集，并对其中的每一个数据集中的数据生成Light curve 对象并进行预处理
def load_original_data_K_fold(K=10, GP_limit=2500):
    for i in range(5, K):
        train_names, val_names, test_names = K_fold_split(K, fold=i)
        print('loading data %dfold No%d'%(K,i))
        train_data = [load_data(train_names[j], GP_model=True if len(train_names[j]) <= GP_limit else False) for j in range(0,11)]
        #预先拟合好GP_model可以加快bagging所需的重采样过程（不用反复拟合）
        #依然有重复计算，K折还是重复拟合了K倍，还可以优化一波，不过好像改起来比价麻烦，还是用计算机时间来节省我的时间吧
        #修改计划有，在util里，有需要再继续吧
        val_data = [load_data(val_names[j], GP_model=True) for j in range(0,11)]
        test_data = [load_data(test_names[j], GP_model=False) for j in range(0,11)]
        original_dataset = (train_data, val_data, test_data)
        f = open('data/original_dataset_%dfold_No%d'%(K,i), 'wb')
        pickle.dump(original_dataset, f)
        f.close()



from util import create_dataset
import os

# 定义create_dataset_K_fold函数，用于将之前的K折数据集进行数据增强
def create_dataset_K_fold(K=10):
    for i in range(0, K):
        original_dataset = 'data/original_dataset_%dfold_No%d'%(K,i)
        for bag in range(0,10):
            print('generating data fold_No%d bag No%d'%(i,bag))
            create_dataset(original_dataset,2500, down_sample=True, instance=bag,
                aug_val=True)




# 调用定义的函数 ，开始生成K折数据集
load_original_data_K_fold()

# 调用定义的函数，开始生成K折数据集的数据增强
create_dataset_K_fold()



# 生成K折数据集的图像数据增强
from util import create_dataset
original_dataset = 'data/original_dataset_0.60_0.30_0.10'
create_dataset(original_dataset, 3000, prefix='_uncertainty_image', down_sample=True, aug_val=True, image=True)