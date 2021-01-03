
from Code.Dataloader import batch_loader
from Code.VAE import VariationalInference,VariationalAutoencoder
from Code.MNEplotter import *
from collections import defaultdict

import numpy as np
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(32)
class VEA_experiment():
    def __init__(self,Data_loader_train,Data_loader_validation,batch_size,latent_features,n_batches,beta,input_shape=[19,250]):

        #setup hyperparameters
        self.batch_size=batch_size
        self.n_batches=n_batches

        self.write_log=False


        #Load dataset
        self.train_set=Data_loader_train
        self.validation_set=Data_loader_validation


        #Setup model
        self.vi = VariationalInference(beta=beta)
        self.VAE= VariationalAutoencoder(self.vi,input_shape, latent_features).to(device)

        # define dictionary to store the training curves
        self.train_history = defaultdict(list)
        self.validation_history = defaultdict(list)

        self.optimizer = torch.optim.Adam(self.VAE.parameters(), lr=1e-3)

    def swap_nn(self,encoder,decoder,flaten=True):
        """
        Takes two nerural nets, in the format nn.Sequential
        and swap them with the autoencoders orignal encoder and decoder

        If using an 2dCNN set flatten = false
        """
        self.VAE.flaten=flaten
        self.VAE.encoder=encoder.to(device)
        self.VAE.decoder=decoder.to(device)
        if self.write_log:
            with open(self.file_name, 'a') as file:
                file.write(str(self.VAE))
                file.write("\n")
        else:
            print(self.VAE)

    def make_logfile(self,file_name,init_message=""):
        self.write_log=True
        self.file_name=file_name
        with open(self.file_name,'w') as file:
            file.write(init_message+"\n")
        with open(self.file_name, 'a') as file:
            file.write(f">> Using device: {torch.cuda.get_device_name(torch.cuda.current_device())} \n")




    def run(self,num_epochs):


        epoch = 0

        while epoch < num_epochs:
            if epoch % 10 == 0:
                if self.write_log:
                    with open(self.file_name,'a') as file:
                        file.write("Epoch {0}/{1} \n".format(epoch, num_epochs))

                else:
                    print("Epoch {0}/{1}".format(epoch, num_epochs))

            epoch += 1

            # gather data for the full epoch
            training_epoch_data = self.train()
            for k, v in training_epoch_data.items():
                self.train_history[k] += [np.mean(training_epoch_data[k])]

            self.test()

    def plot_history(self,path):
        plot_AC(self.train_history, self.validation_history,path,show=False)

    def plot_latents(self,path):
        #ToDO fix this function and standise y's format.
        x, y, _ = self.validation_set.load_all()
        loss, diagnostics, outputs = self.vi(self.VAE, x)

        #encode to number
        Y=torch.Tensor(np.array([(yi == 1).nonzero()[0][0] for yi in y])).type(torch.int)
        plot_2d_latents(outputs, Y, tmp_img=path, show=False)

    def train(self):
        training_epoch_data = defaultdict(list)
        self.VAE.train()

        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for batch in range(self.n_batches):
            x, _,_ = self.train_set.load_random(self.batch_size)

            # perform a forward pass
            # through the model and compute the ELBO
            loss, diagnostics, self.outputs = self.vi(self.VAE, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # gather data for the current bach
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]

        return training_epoch_data


    def test(self):

        # Evaluate on a single batch, do not propagate gradients
        with torch.no_grad():
            self.VAE.eval()

            # Just load a single batch from the test loader
            x, y, _ = self.validation_set.load_all()

            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = self.vi(self.VAE, x)

            # gather data for the validation step
            for k, v in diagnostics.items():
                self.validation_history[k] += [v.mean().item()]

    def show_plots(self,size):
        with torch.no_grad():
            self.VAE.eval()

            # Just load a single batch from the test loader
            x, y, _ = self.validation_set.load_random(size)
            x_new=self.VAE.sample(x)
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = self.vi(self.VAE, x)

            plot_2d_latents(outputs,y)







