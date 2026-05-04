import data_w
from data_w import widedata
from data_w import*
import model_w
from model_w import*
from model_w import WideCNN
#datset=widedata(ligand_path, protein_path,keys,motif_path,affinity_path)


dataset = widedata(ligand_path, protein_path,keys,motif_path,affinity_path)
train_loader, test_loader = load_splitset(dataset, .2)
modelw=WideCNN()

class rmsloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self,yhat,y):
       return torch.sqrt(self.mse(yhat,y))

def train_w(model,train):
    epochs=3
    criterion = rmsloss()
    optimizer=optim.Adam(model.parameters(),lr=0.003)
    for epoch in range(epochs):
        outw = []
        losw = []
        k=0
        for i,j in train:
           m = i[0]
           #m = m.reshape((1, 10, 10)) # For KIBA dataset
           m = m.reshape((1, 141, 101)) # For Davis dataset
           p = i[1]
           #p = p.reshape((1, 6729, 594)) # For KIBA dataset
           p = p.reshape((1, 9552, 467)) # For Davis dataset
           mt = i[2]
           #mt = mt.reshape((1, 1076, 32)) # For KIBA dataset
           mt = mt.reshape((1, 2017, 96)) # For Davis dataset
           out = model(p, m, mt)
           outw.append(out)

           # For Davis dataset
           # Ensure the output and target have the same shape
           out = out.squeeze()  # Remove dimensions of size 1 if necessary
           j = j.squeeze()  # Remove any extra dimensions from the target

           optimizer.zero_grad()
           loss = criterion(out, j)
           losw.append(loss)
           loss.backward()
           optimizer.step()
           k += 1
           if k == 500:
               break
    return outw,losw
trainwide=train_w(modelw,train_loader)
torch.save(modelw,'wide.pt')

if __name__ == '__main__':
   print(trainwide[0])
   print(trainwide[1])








