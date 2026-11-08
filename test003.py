from kan import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(width = [2,1,1],grid = 3,k = 3,seed = 2,device = device)
f = lambda x: torch.exp(torch.special.bessel_j0(20*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f,n_var = 2,train_num = 1500,device = device)
model.fit(dataset,opt = "LBFGS",steps = 50,stop_grid_update_step=20)
model.suggest_symbolic(0,0,0)
model.plot()
plt.show()