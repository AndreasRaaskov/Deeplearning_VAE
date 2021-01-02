
import sys,torch
from Code.Experiment_setup import VEA_experiment
from torch import nn

# setup device
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

save_path=r"CNN1/"
Experiment_name=r"CNN1"

Data_train_path=r"C:\Users\Andre\Desktop\Deeplearning local\DeepLearning-VAE\Data\Fourier_0_overlab_train"
Data_validation_path=r"C:\Users\Andre\Desktop\Deeplearning local\DeepLearning-VAE\Data\Fourier_0_overlab_validation"
batch_size=32
n_batches=10  #How many batches for each epochs
latent_features=100 #Number of dimensions on latent layer
n_epoke=100


encoder = nn.Sequential(
    nn.Conv1d(in_channels=19,out_channels=256,padding=5,kernel_size=11),
    nn.ReLU(),
    nn.Conv1d(in_channels=256,out_channels=1,padding=5,kernel_size=11),
    nn.ReLU(),
    nn.Linear(in_features=250, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
    nn.Linear(in_features=128, out_features=2 * latent_features)  # <- note the 2*latent_features
)

# Generative Model
# Decode the latent sample `z` into the parameters of the observation model
# `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
decoder = nn.Sequential(
    nn.Linear(in_features=latent_features, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=250),
    nn.ReLU(),
    #nn.Linear(in_features=256, out_features=2*4750)
    nn.Conv1d(in_channels=1, out_channels=256, padding=5, kernel_size=11),
    nn.ReLU(),
    nn.Conv1d(in_channels=256, out_channels=19*2, padding=5, kernel_size=11),
)

EX=VEA_experiment(Data_train_path=Data_train_path,Data_validation_path=Data_validation_path,batch_size=batch_size,
                  n_batches=n_batches,latent_features=latent_features)
EX.make_logfile(save_path+Experiment_name+"_log.txt","Test")
EX.swap_nn(encoder,decoder,flaten=False)
EX.run(num_epochs=n_epoke)
torch.save(EX.VAE.state_dict(), save_path+Experiment_name+"_model.pt")
EX.plot_history(save_path+Experiment_name+"_history.png")
