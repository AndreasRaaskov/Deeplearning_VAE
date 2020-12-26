
import sys,torch
from Code.Experiment_setup import VEA_experiment
from torch import nn

# setup device
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

save_path=r"Dense1/"
Experiment_name=r"Dense1"

Data_train_path=r"C:\Users\Andre\Desktop\Deeplearning local\DeepLearning-VAE\Data\Fourier_0_overlab_train"
Data_validation_path=r"C:\Users\Andre\Desktop\Deeplearning local\DeepLearning-VAE\Data\Fourier_0_overlab_validation"
batch_size=32
n_batches=10  #How many batches for each epochs
latent_features=100 #Number of dimensions on latent layer
n_epoke=1000




EX=VEA_experiment(Data_train_path=Data_train_path,Data_validation_path=Data_validation_path,batch_size=batch_size,
                  n_batches=n_batches,latent_features=latent_features)
EX.make_logfile(save_path+Experiment_name+"_log.txt","Experiment with auto encoder used for presentation but with 100 latent features")
EX.run(num_epochs=n_epoke)
torch.save(EX.VAE.state_dict(), save_path+Experiment_name+"_model.pt")
EX.plot_history(save_path+Experiment_name+"_history.png")

