#Qiuqk test of bug appering for certain files.
from MakeDict import findEDF
from Dataloader import dataloader,preprossing

#Part to bad file
data_path=r"C:\Users\Andre\Desktop\Deeplearning local\artifact_dataset\artifact_dataset"

loader=preprossing(Time_interval=1,Overlap=0.75,Data_paht=data_path,Fourier=True)
edfDict=findEDF(DataDir=data_path)
loader.auto_TrainTest_split(edfDict,"min",file_name="Fourier_0.75_overlab",test_size=32)
