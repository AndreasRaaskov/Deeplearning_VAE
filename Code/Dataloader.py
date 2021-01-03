
import numpy as np

import os, mne, re
from datetime import datetime
from mne.io import read_raw_edf
from collections import defaultdict
path=os.path.join(os.getcwd(),r"Data\s007_2012_07_25\00001154_s007_t000")
import pandas as pd
import pickle
import torch
from scipy.fft import fft

class dataloader:
    def __init__(self,Time_interval,Overlap,Fourier=False,Data_paht=None,freq=250,Chanels=None):
        """
        tWindow is how long each window is in seconds
        Overlap given as a float eg: 0.25 for 0.25% overlap
        """
        self.one_hot_eoncoding=np.array(["null","eyem","chew","shiv","elpp","musc"])
        self.freq=freq
        self.tWindow=Time_interval
        self.Overlap=Overlap
        self.Fourier=Fourier
        if Chanels==None:
            self.CH_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5','T6','Fz', 'Cz', 'Pz']
        else:
            self.CH_names=Chanels


        if Data_paht is None:
            self.DataDir=os.getcwd()
        else:
            self.DataDir=Data_paht

    def loadDict(self,edfDict,index=[None]):
        windowDict = defaultdict()
        keys=edfDict.keys()
        if index!=[None]:
            keys=index
        else:
            keys = edfDict.keys()
        for idx in keys:
            edfDict[idx] = self.readRawEdf(edfDict[idx])
            windowDict[idx] = self.slidingWindow(edfDict[idx])
        return windowDict

    def loadBatch(self,edfDict,windowlist,filelist):
        batch={}
        X=[]
        Y=[]
        n=0
        for i,idx in enumerate(windowlist):
            edfDict[idx] = self.readRawEdf(edfDict[idx])
            windows=self.slidingWindow(edfDict[idx])
            for w in filelist[i]:
                batch[n]=windows[f"window_{w}"]
                batch[n]['info']={"File index": i, "window index": w}
                X.append(windows[f"window_{w}"]['X'])
                Y.append(windows[f"window_{w}"]['Y'])
                n+=1
        return batch,X,Y


    def loadFile(self,path):
        pass

    def anno_mapping(self,edfDict,index=[None]):
        if index!=[None]:
            keys=index
        else:
            keys = edfDict.keys()
        annothlist=[[] for i in self.one_hot_eoncoding]
        lablelist=list(self.one_hot_eoncoding)
        for key in keys: #Go over each file.
            anno=self.annoTUH(self.DataDir+"/"+edfDict[key]["path"]+r".tse")
            tN=int(anno.iloc[-1,1]) #Get recording endtine
            t0=int(anno.iloc[0,0]) #Get startime
            file_overwiew=[]
            step=int(self.freq*self.tWindow*(1-self.Overlap))/self.freq #atemt to alligne with slidinger window.
            for i,t in enumerate(np.arange(t0, tN, step)): #Go over the file.
                tEnd=t+self.tWindow
                iStart = sum(anno[0] <= t) - 1
                iEnd = sum(anno[1] <= tEnd)
                if iStart == iEnd:
                    # Assing lable as last started artifact.
                    lable = anno.iloc[iStart, 2]
                else:
                    # if and artifact end in the window assign lable to domenet artifact
                    lable = anno.iloc[[iStart, iEnd][np.argmax([t - anno.iloc[iStart, 0], tEnd - anno.iloc[iStart, 1]])], 2]

                lableidx=lablelist.index(lable)
                annothlist[lableidx].append([key,i])
                file_overwiew.append(lable)
                edfDict[key]["lables"]=file_overwiew






        return edfDict,annothlist

    def readRawEdf(self,edfDict=None,read_raw_edf_param={'preload':True, "stim_channel":"auto"}):
        """
        Function to read the edf file in one given location.
        """
        #EEGseries= read_raw_edf(self.DataDir+"/"+edfDict["path"]+".edf", **read_raw_edf_param)
        EEGseries=read_raw_edf(self.DataDir+"/"+edfDict["path"]+".edf",preload= True,verbose="WARNING")
        #EEGseries.plot()
        EEGseries.resample(sfreq=self.freq)  # Down sample to desired frequense
        EEGseries=self.TUHfooDef(EEGseries)
        EEGseries.pick_channels(self.CH_names)
        EEGseries.set_montage(mne.channels.make_standard_montage(kind="standard_1005", head_size=0.095))
        #print(EEGseries.info["ch_names"])
        #print(len(EEGseries.info["ch_names"]))
        EEGseries.set_eeg_reference() #Using avage as reference 
        edfDict["rawData"] =EEGseries
        edfDict["Annotation"] = self.annoTUH(self.DataDir+"/"+edfDict["path"]+r".tse")
        #EEGseries.plot()

        # comment this out to get meta data on recording time stamps, WARNING will given and error in python 3.7
        # tStart = edfDict["rawData"].annotations.orig_time-timedelta(hours=1)
        # tLast = int((1+edfDict["rawData"].last_samp)/edfDict["rawData"].info["sfreq"])
        # edfDict["t0"] = tStart
        # edfDict["tN"] = tStart + timedelta(seconds=tLast)

        if self.CH_names != EEGseries.info["ch_names"]:
            raise Exception(f"Montage Error")
        return edfDict

    def TUHfooDef(self,EEGseries=False):
        """
        Gets the chanels reigt
        """
        for i in EEGseries.info["ch_names"]:
            #Regular expretion to detect EGG "CH" -REF
            reREF = r"(?<=EEG )(.*)(?=-REF)"

            #Detect EEG -LE
            reLE=r"(?<=EEG )(.*)(?=-LE)"

            #Chanels where the seocnd letter need to be lower key.
            reLowC = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ','T','A']

            #Clean chanel names -REF
            if re.search(reREF, i) and re.search(reREF, i).group() in reLowC:
                lowC = i[0:5] + i[5].lower() + i[6:]
                mne.channels.rename_channels(EEGseries.info, {i: re.findall(reREF, lowC)[0]})
            elif re.search(reREF, i):
                mne.channels.rename_channels(EEGseries.info, {i: re.findall(reREF, i)[0]})

            #Clean -Le
            elif re.search(reLE, i) and re.search(reLE, i).group() in reLowC:
                lowC = i[0:5] + i[5].lower() + i[6:]
                mne.channels.rename_channels(EEGseries.info, {i: re.findall(reLE, lowC)[0]})
            elif re.search(reLE, i):
                mne.channels.rename_channels(EEGseries.info, {i: re.findall(reLE, i)[0]})
            else:
                pass
                #print not clean channels
                #print(i)
        return EEGseries

    def annoTUH(self,annoPath=False, header=None):
        df = pd.read_csv(annoPath, sep=" ", skiprows=1, header=header)
        df.fillna('null', inplace=True)
        #df.dropna()
        #annoTUH = df[df[0].between(window[0], window[1]) | df[1].between(window[0], window[1])]
        return df

    def slidingWindow(self,edfInfo):
        """
        Split singal up into windows.
        """
        windowEEG = defaultdict(list)
        sampleWindow = self.tWindow*self.freq
        tN = edfInfo["rawData"].last_samp
        steps=int(sampleWindow*(1-self.Overlap))
        n=0
        for i in range(0, tN, steps):
            windowKey = f"window_{n}"

            windowEEG[windowKey] = self.DataMaker(edfInfo, t0=i, tWindow=sampleWindow)
            n+=1
        if (1+tN) % int(steps) != 0:
            pass #Trow away the last window comment out to get it back
            #windowKey = f"window_{n}"
            #windowEEG[windowKey] = self.DataMaker(edfInfo, t0=int(tN-sampleWindow), tWindow=sampleWindow)
        return windowEEG

    def DataMaker(self,EEGseries, t0, tWindow):
        """
        Extract data for a and lables for a window. If fourier transformations is neede this is the place to do it.
        """
        tStart=t0/self.freq
        tEnd=(t0+tWindow)/self.freq
        Data={}
        Data["stepval"] = [t0,t0+tWindow]
        Data["tval"] = [tStart,tEnd]
        Data["X"] = EEGseries["rawData"].get_data(start=int(t0), stop=int(t0+tWindow))

        if self.Fourier:
            #apply fourier transformations and take abselut value
            Data["X"]=[abs(x) for x in fft(Data["X"])]

        df=EEGseries["Annotation"]
        #Data["Y"]= df.loc[df[0].between(tStart, tEnd) | df[1].between(tStart, tEnd)][2]
        iStart=sum(df[0]<=tStart)-1
        iEnd=sum(df[1]<=tEnd)
        if iStart==iEnd:
            #Assing lable as last started artifact.
            lable=df.iloc[iStart,2]
        else:
            #if and artifact end in the window assign lable to domenet artifact
            lable=df.iloc[[iStart,iEnd][np.argmax([tStart-df.iloc[iStart,0],tEnd-df.iloc[iStart,1]])],2]
        Data["lable"]=lable
        Data["Y"]=self.one_hot_eoncoding==lable
        if np.sum(Data["Y"]) !=1:
            raise Exception("To few or to many labels.")
        return Data

class preprossing(dataloader):
    def auto_dataset(self,edfDict,class_size,file_name):
#        edfDict,annothlist=self.anno_mapping(edfDict)
        annothlist=np.load('testIDX.npy', allow_pickle=True)
        windowlist, filelist=self.make_batch(annothlist,class_size)
        batch,x,y=self.loadBatch(edfDict,windowlist,filelist)
        self.save_batch(batch,file_name)

    def auto_TrainTest_split(self,edfDict,class_size,file_name,test_ratio=None,test_size=None):
        """
        Like auto dataset but makes train test split
        test ration float between 1 and 0
        test size int must be dividable by 6
        "Either test ratio or test size must be none"
        """

        edfDict, annothlist = self.anno_mapping(edfDict)

        if test_size is None:
            size_per_class = np.min([len(annothlist[i]) for i in range(len(annothlist))])
            n_size=int(size_per_class*test_ratio*len(annothlist))
        elif test_ratio is None:
            n_size=test_size
        else:
            raise print("Either test ratio or test size must be none")

        #make test set.
        test_windowlist, test_filelist=self.make_batch(annothlist,int(n_size/6))

        batch,x,y=self.loadBatch(edfDict,test_windowlist,test_filelist)
        self.save_batch(batch,file_name+"_validation")

        #Window usede in test set
        annothlist=annothlist
        for i,lable in enumerate(annothlist):
            for n,a in enumerate(lable):
                for j, file in enumerate(test_windowlist):
                    if a[0]==file:
                        for window in test_filelist[j]:
                            if a[1]==window:
                                annothlist[i].remove([file,window])


        # make train set
        train_windowlist, train_filelist = self.make_batch(annothlist, class_size)
        batch, x, y = self.loadBatch(edfDict, train_windowlist, train_filelist)
        self.save_batch(batch,file_name+"_train")


    def save_batch(self,batch,file_name):
        pickle.dump(batch,open(file_name,'wb'))

    def make_batch(self,IDXlist, size_per_class, Nlable=6,replace=False):
        """
        make balance dataset, sampling with replacement.
        """
        filelist = []
        windowlist = []

        if size_per_class=="min":
            size_per_class=np.min([len(IDXlist[i]) for i in range(Nlable)])

        for i in range(Nlable):
            elements = np.random.choice(len(IDXlist[i]),size_per_class, replace=replace)
            for e in elements:
                window = int(IDXlist[i][e][0])
                try:  # See if window already is in list else append it
                    winidx = windowlist.index(window)
                    filelist[winidx].append(int(IDXlist[i][e][1]))
                except ValueError:
                    windowlist.append(window)
                    filelist.append([int(IDXlist[i][e][1])])

        return windowlist, filelist

class batch_loader():
    def __init__(self,path,device="CPU"):
        self.path=path
        self.pre_loaded=False
        self.device=device

    def pre_load(self):
        self.pre_loaded=True
        self.data=pickle.load(open(self.path,'rb'))
        self.data_size=len(self.data.keys())
        self.lable_list=[]
        for n in range(self.data_size):
            self.lable_list.append(np.argmax(self.data[n]['Y']))


    def load(self,idx,encoded=True):
        X=[]
        Y=[]
        meta=[]
        if self.pre_loaded:
            for key in idx:
                X.append(self.data[key]['X'])
                meta.append(self.data[key]['info'])
                if encoded:
                    Y.append(self.data[key]['Y'])
                else:
                    Y.append(self.data[key]['lable'])
        X = torch.Tensor(X)
        if encoded:
            Y = torch.Tensor(Y)
            return X.to(self.device),Y.to(self.device),meta
        else:
            return X.to(self.device), Y, meta

    def load_random(self,size,encoded=True):
        batch = np.random.choice(self.data_size, size)
        X,Y,meta=self.load(batch,encoded)

        return X,Y,meta

    def load_all(self,encoded=True):
        idx=list(self.data.keys())
        X,Y,meta=self.load(idx,encoded)
        return X,Y,meta




#loader=dataloader(Time_interval=1,Overlap=0)
#edfdict={"path": path}
#dataDict=loader.readRawEdf(edfdict)
#windowdict=loader.slidingWindow(dataDict)
#print(windowdict)