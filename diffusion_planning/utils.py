import copy
import numpy as np
from colorama import Fore
import torch
import torch.nn as nn
import math

# ----- OGBench ------

def transforms_dataset_ogbench(raw_dataset):
    current_trajectory = [[], []]
    dataset = []
    num_trajs = 0
    for i in range(len(raw_dataset['valids'])):
        if(raw_dataset['valids'][i] == 0):
            if(num_trajs%1000 == 0):
                print("done")
            current_trajectory[0].append(raw_dataset["observations"][i])
            dataset.append([np.array(current_trajectory[0]), np.array(current_trajectory[1])])
            current_trajectory = [[], []]
            num_trajs += 1
        else:
            current_trajectory[0].append(raw_dataset["observations"][i])
            current_trajectory[1].append(raw_dataset["actions"][i])
    return dataset, dataset[0][0].shape[-1], dataset[0][1].shape[-1]

# ----- Rendering -----

def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)

# ----- Parameters ------

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'
    
def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def report_parameters(model, topk=10):
	counts = {k: p.numel() for k, p in model.named_parameters()}
	n_parameters = sum(counts.values())
	print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

	modules = dict(model.named_modules())
	sorted_keys = sorted(counts, key=lambda x: -counts[x])
	max_length = max([len(k) for k in sorted_keys])
	for i in range(topk):
		key = sorted_keys[i]
		count = counts[key]
		module = param_to_module(key)
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters

# ----- Meta training algorithms -----

class EMA():
    def __init__(self, beta, model):
        self.beta = beta
        self.state_dict = copy.deepcopy(model.state_dict())

    def step(self, model):
        model_state_dict = model.state_dict()
        for key in self.state_dict.keys():
            self.state_dict[key] = self.state_dict[key]*self.beta+(1-self.beta)*model_state_dict[key]

# ----- Generic AI models -----

class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list,
                 output_dim, ## int or None 
                 activate_final, 
                 n_dpout_until=2,
                 act_f=nn.GELU,
                 prob_dpout=0.2
                 ):
        '''
        This is just a helper MLP model from the OGBench GC Policy
        Args:
            hidden_dims (list): [512, 512, 512] or [in, 256, 256, out]
            final_fc_init_scale: 1e-2
        '''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        module_list = []
        
        # final_fc_init_scale = mlp_config['final_fc_init_scale']

        if output_dim is None:
            layer_dim = [self.input_dim,] + hidden_dims
            self.output_dim = hidden_dims[-1]
        else:
            # (in, 512, 256, 128, out) ? out of date
            layer_dim = [self.input_dim,] + hidden_dims + [output_dim,]
        
        ## Jan 8 NEW, default: the last 2 layers do not have dropout
        assert n_dpout_until in [2,1,0]

        num_layer = len(layer_dim) - 1

        for i_l in range(num_layer):
            tmp_linear = nn.Linear(layer_dim[i_l], layer_dim[i_l+1])
            nn.init.zeros_(tmp_linear.bias)

            module_list.append( tmp_linear )
            module_list.append( act_f() )

            if i_l < num_layer - n_dpout_until:
                # assert False, 'bug free, but not used.'
                module_list.append( nn.Dropout(p=prob_dpout), )

        # pdb.set_trace()
        if not activate_final:
            del module_list[-1] # no relu at last
        
        self.encoder = nn.Sequential(*module_list)
        self.n_dpout_until = n_dpout_until
        
        # from diffuser.utils import print_color
        # print_color(f'[MLP_InvDyn_OgB_V3]  {num_layer=}, {layer_dim=}')
            
    def forward(self, x):
        x = self.encoder(x)
        return x

# ----- Generic generative AI models -----

class FullDiffusion(nn.Module):
    def __init__(self, beta_0, beta_1, model, *args, **kwargs):
        super().__init__()
        self.log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1 - beta_0)
        self.dtlog_alpha = lambda t: -0.5*beta_0-0.5*t*(beta_1 - beta_0)
        self.sigma = lambda t: t
        self.beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0))

        self.score_model = model(*args, **kwargs)
        self.t = None

    def add_noise(self, data, Nt=100):
        """
        data : (*, )

        Returns : _ X_t : (Nt, *)
        _ noise : (N_t, *)
        """
        self.t = torch.rand(Nt, *[1 for i in range(len(data.shape))], device=data.device)
        eps = torch.randn(Nt, *data.shape, device=data.device)

        data = data[None,...]*torch.exp(self.log_alpha(self.t))+self.sigma(self.t)*eps
        return (data, eps)

    def compute_loss(self, tuple_noisy, cond_data=None):
        """
        data : (*, D)
        """
        data, eps = tuple_noisy

        preds = None
        if cond_data is None:
            preds = self.score_model(self.t, data)
        else:
            preds = self.score_model(self.t, data, cond_data)

        #print(eps.shape, preds.shape)
        loss = torch.mean(torch.sum((eps+preds)**2, dim=-1))
        #model_loss = self.score_model.compute_loss()
        #print(loss, model_loss)
        return loss
    
    def step(self, X_t, t, T):
        dt = 1/T
        return X_t - dt*(self.dtlog_alpha(t)*X_t - 2*self.beta(t)*self.score_model(t*torch.ones(1, 1, 1), X_t)) + np.sqrt(2*self.sigma(t)*self.beta(t)*dt)*torch.randn(*X_t.shape, device=X_t.device)
    
    def generate_samples(self, T, B, S):
        timesteps = np.flip(np.linspace(0.0, 1.0, T), 0)
        X_t, dt = torch.randn(1, B, S), 1/T
        for t in timesteps:
            X_t = X_t - dt*(self.dtlog_alpha(t)*X_t - 2*self.beta(t)*self.score_model(t*torch.ones(1, 1, 1), X_t)) + np.sqrt(2*self.sigma(t)*self.beta(t)*dt)*torch.randn(*X_t.shape, device=X_t.device)
        return X_t.detach().numpy()
    
# ----- computing tools -----
def patch_size_step(number_patches, length):
     step = math.ceil(length/number_patches)-1
     if step <= 0:
        raise ZeroDivisionError
     return length-(number_patches-1)*step, step 