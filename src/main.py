import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.nn.utils as utils 
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms 
import numpy as np 
from tqdm import tqdm 


"""
now with no mish

gives hard noisy boundries when no random fourier features are used

colors seem less bright/diverse and the shapes sharper (what in this case is nice) when random fourier features are used
"""

class random_fourier_encoding(nn.Module):
    def __init__(self,components=40,dim=2,std =75,gpu=True):
        super().__init__()
        self.B = torch.randn(dim,components) * std 
        
        if(gpu):
            self.B = self.B.cuda()
    
    
    def forward(self,x):
    
        return torch.sin(torch.matmul(x,self.B))


class network(nn.Module):
    def __init__(self,input_size=100,hidden_size=256,output_size=3,with_random_fourier=True):
        super().__init__()
        self.with_random_fourier = with_random_fourier
        if(self.with_random_fourier):
            self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size),nn.Sigmoid())
        else:
            self.layers = nn.Sequential(nn.Linear(2,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size),nn.Sigmoid())
  
        self.fourier_enc = random_fourier_encoding(components=input_size)
    def forward(self,x):
        if(self.with_random_fourier):
            x = self.fourier_enc(x)
        return self.layers(x)

    def to_image(self,output_batch,N):
        red = output_batch[:,0].view(1,N,N)
        green = output_batch[:,1].view(1,N,N)
        blue = output_batch[:,2].view(1,N,N)
        return torch.cat((red,green,blue),dim=0)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    
    image = Image.open("fox.jpg") 
    N = 200
    image = image.resize((N,N))
    
    target_image = transforms.ToTensor()(image).cuda()
    target_image = torch.cat((target_image[0].view(-1,1),target_image[1].view(-1,1),target_image[2].view(-1,1)),dim=1)
    plt.imshow(image)
    plt.show()
    x = np.arange(0,N)
    y =np.arange(0,N)
    xx, yy = np.meshgrid(x,y)
    xx = torch.tensor(xx,dtype=torch.float).view(-1)
    yy = torch.tensor(yy,dtype=torch.float).view(-1)

    input_image = torch.tensor(list(zip(yy,xx))).cuda()/N
    


    net = network().cuda()
    optimizer = optim.Adam(net.parameters(),lr=0.01)

    epochs = 2000
    plt.ion()
    for i in range(epochs):
        optimizer.zero_grad()
        

        indexes = torch.tensor(np.random.choice(np.arange(len(input_image)),100),dtype=torch.int64).cuda()
        
        output = net(torch.index_select(input_image, 0, indexes))
        loss = nn.MSELoss()(output,torch.index_select(target_image, 0, indexes))

        loss.backward()
        print(loss)
        optimizer.step()


        with torch.no_grad():
            output = net.to_image(net(input_image),N)
            plt.cla()
            plt.imshow(transforms.ToPILImage()(output.detach()))
            plt.pause(0.01)
            
    plt.show()
    plt.pause(1000)
    plt.waitforbuttonpress()
