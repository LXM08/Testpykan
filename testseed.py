from kan import *
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)
seedlist = []
for i in range(1,10):
    seedlist.append(i)
f = lambda x: torch.exp(torch.special.bessel_j0(20*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f,n_var = 2,train_num = 1500,device = device)
train_rmse = []
test_rmse = []
for i in range(9):
    model = KAN(width = [2,1,1],grid = 3,k = 3,seed = seedlist[i],device = device)
    res = model.fit(dataset,opt = "LBFGS",steps = 50,stop_grid_update_step=20)
    train_rmse.append(res['train_loss'][-1].item())
    test_rmse.append(res['test_loss'][-1].item())
plt.plot(seedlist,train_rmse,marker = 'o')
plt.plot(seedlist,test_rmse,marker = 'o')
plt.ylabel("rmse")
plt.xlabel("seeds")
plt.legend(loc = "lower left")
plt.show()