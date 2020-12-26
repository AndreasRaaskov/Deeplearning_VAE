import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.display import Image, display, clear_output
from sklearn.manifold import TSNE
from torch import Tensor
from torch.distributions import Normal
from torchvision.utils import make_grid

def plot_elbo(training_data,validation_data):
    # plot ELBO
    plt.figure()
    plt.title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
    plt.plot(training_data['elbo'], label='Training')
    plt.plot(validation_data['elbo'], label='Validation')
    plt.legend()

def plot_kl(training_data,validation_data):
    # plot KL
    plt.figure()
    plt.title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
    plt.plot(training_data['kl'], label='Training')
    plt.plot(validation_data['kl'], label='Validation')
    plt.legend()

def plot_nll(training_data,validation_data):
    # plot NLL
    plt.figure()
    plt.title(r'$\log p_\theta(\mathbf{x} | \mathbf{z})$')
    plt.plot(training_data['log_px'], label='Training')
    plt.plot(validation_data['log_px'], label='Validation')
    plt.legend()
    
def plot_samples(x, title):
    channels = 19
    fig, axs = plt.subplots(channels,figsize=(8,22))
    t = x[0]
    plt.title(title)

    for i in range(channels):
        axs[i].plot(t[i])
    
def plot_posteriors(outputs):
    # plot posterior samples
    plt.figure()
    title = 'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$'
    px = outputs['px']
    x_sample = px.sample().to('cpu')
    plot_samples(x_sample, title)
    
def plot_prior(vae,x):   
    # plot prior samples
    title='Samples $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim p(\mathbf{z})$'
    px = vae.sample_from_prior(batch_size=x.size(0))['px']
    x_samples = px.sample()
    plot_samples(x_samples, title)

    
def plot_latent_space(outputs, y):
    fig, axes = plt.subplots(1, figsize=(10, 10), squeeze=False)
    plot_2d_latents(axes[0,0], outputs, y)

def plot_2d_latents(ax, outputs, y):
    z = outputs['z']
    qz = outputs['qz']
    z = z.to('cpu')
    y = y.to('cpu')
    scale_factor = 2
    batch_size = z.shape[0]
    palette = sns.color_palette()
    colors = [palette[l] for l in y]

    # plot prior
    prior = plt.Circle((0, 0), scale_factor, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior)

    # plot data points
    mus, sigmas = qz.mu.to('cpu'), qz.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(z[:, 0], z[:, 1], color=colors)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    
    ax.set_aspect('equal', 'box')
    ax.set_title('Latent space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')


def plot_interpolations(vae):
    device = next(iter(vae.parameters())).device
    nrow = 10
    nsteps = 10
    prior_params = vae.prior_params.expand(2 * nrow, *vae.prior_params.shape[-1:])
    mu, log_sigma = prior_params.chunk(2, dim=-1)
    pz = Normal(mu, log_sigma.exp())
    z = pz.sample().view(nrow, 2, -1)
    t = torch.linspace(0, 1, 10, device=device)
    zs = t[None, :, None] * z[:, 0, None, :] + (1 - t[None, :, None]) * z[:, 1, None, :]
    px = vae.observation_model(zs.view(nrow * nsteps, -1))
    x = px.sample()
    x = x.to('cpu')
    plot_samples(x)
