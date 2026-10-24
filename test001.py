from kan import *
if(torch.cuda.is_available() == True):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
model = KAN(width=[3,1,1], grid=3, k=3, seed=1, device=device)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2 + x[:,[2]])
dataset = create_dataset(f, n_var=3, device=device)