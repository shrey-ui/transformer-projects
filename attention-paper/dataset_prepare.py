from datasets import inspect_dataset, load_dataset_builder
import datasets
from datasets import load_dataset
import pandas as pd

def save_dataset(path):
    builder= load_dataset_builder(
            path,
            language_pair= ("de", "en"),
            subsets= {
                datasets.Split.TRAIN: ['europarl_v7'],
                datasets.Split.VALIDATION: ['euelections_dev2019']
                },
            )

    builder.download_and_prepare()
    ds= builder.as_dataset()
    

    ds.save_to_disk("./dataset/train-translate")

    return ds

if __name__ == "__main__":
    #ds= save_dataset("./dataset/wmt_utils.py")
    #data_files= {"train" : "data-00000-of-00002.arrow"}
    raw_train, raw_test = load_dataset('./dataset/wmt14', "de-en", split= ["train", "test[:3003]"])
    #raw.to_csv('./dataset/wmt14/')
    #print(raw.data)
    #print(raw.shape)
    print(type(raw_train))
    dataset= pd.DataFrame(raw_train)
    dataset_test= pd.DataFrame(raw_test)

    dataset.to_csv('./dataset/wmt14/dataset_init_train.csv')
    dataset_test.to_csv('./dataset/wmt14/dataset_init_test.csv')

    
    #df_train= pd.read_csv('./dataset/wmt14/dataset_init_test.csv', encoding= 'utf-8')
    #for ind, rows in df_train.iterrows():
        #print(rows['translation'])
    #print(df_train)

    #print(dataset)

