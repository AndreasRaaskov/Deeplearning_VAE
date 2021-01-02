
from MakeDict import findEDF
from Dataloader import dataloader,preprossing

#Part to bad file
data_path=r"C:\Users\Andre\Desktop\Deeplearning local\artifact_dataset\artifact_dataset"
print("Generating dataset")
loader=preprossing(Time_interval=1,Overlap=0,Data_paht=data_path,Fourier=False)
edfDict=findEDF(DataDir=data_path)
print("Data anotated")
loader.auto_TrainTest_split(edfDict,"min",file_name="0_overlab_70_30_split",test_ratio=0.30)
