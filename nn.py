print('import...',end='')

from PIL import Image
from skimage import io
import numpy as np
#import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F

import random

import torch
import torchvision
print(' finnished')


categories=['RING','NOISE','DOUBLES','BLOB']
print('Categories:',categories)

print('import...')
data=[]
for i in range(len(categories)):
    tiff_file=io.imread(categories[i]+'.tiff')
    data.append(np.array(tiff_file))
    print(categories[i],':',np.array(tiff_file).shape)
print('import success')



class Data_Loader:
    def __init__(self):
        print('data_loader.__init__():')

        #set aside 100 of each category for validation

        self.train_data=[]
        self.valid_data=[]
        self.Cur_train_idx=[]
        self.Cur_train_category=0
        print('\t setting aside validation set:')
        for i in range(len(categories)):
            np.random.shuffle(data[i]) ################################ comment if validation set should remain the same in each training
            self.valid_data.append(np.array(data[i][0:100]))
            self.train_data.append(np.array(data[i][100:-1]))
            print('\t',categories[i],':',self.valid_data[i].shape,self.train_data[i].shape)

            self.Cur_train_idx.append(0)
        
        #prepare the validation set
        self.validation_batches_list=[]
        aux_data=[]
        aux_targets=[]
        for category in range(len(categories)):
            for clip_x in range(4,8+1,2):
                for clip_y in range(4,8+1,2):
                    for transform_type in range(8):
                        for idx in range(len(self.valid_data[category])):
                            Cur_img=self.valid_data[category][idx]
                            Cur_img=Cur_img[clip_x:clip_x+20,clip_y:clip_y+20]
                            if transform_type<4:
                                Cur_img=np.flipud(Cur_img)
                            if transform_type%2==0:
                                Cur_img=np.fliplr(Cur_img)
                            if (transform_type%4) < 2:
                                Cur_img=np.transpose(Cur_img)
                            Cur_img=(Cur_img-np.mean(Cur_img))/np.std(Cur_img)
                            #Cur_img=np.reshape(Cur_img,20*20) ######################## flattening, say for non convolutional network
                            aux_data.append([Cur_img])
                            aux_targets.append(category)
        print('\t\t\t\t aux_data len', np.array(aux_data).shape)
        print('\t\t\t\t aux_targ len', np.array(aux_targets).shape)

        k=0
        while k*128+128<len(aux_data):
            self.validation_batches_list.append((torch.from_numpy(np.array(aux_data[k*128:k*128+128])),torch.tensor(aux_targets[k*128:k*128+128])))
            k+=1
        self.validation_batches_list.append((torch.from_numpy(np.array(aux_data[k*128:-1])),torch.tensor(aux_targets[k*128:-1])))
        print('\t\t\t\t vbll', len(self.validation_batches_list))




    

    def get_validation_batches_list(self):
        return self.validation_batches_list


    def train_batch(self,batch_size):
        ret_data=[]
        ret_targets=[]
        for j in range(batch_size):
            Cur_img=self.train_data[self.Cur_train_category][self.Cur_train_idx[self.Cur_train_category]]
            clip_x=random.randint(4,8)
            clip_y=random.randint(4,8)
            Cur_img=Cur_img[clip_x:clip_x+20,clip_y:clip_y+20]
            if random.random()>0.5:
                Cur_img=np.flipud(Cur_img)
            if random.random()>0.5:
                Cur_img=np.fliplr(Cur_img)
            if random.random()>0.5:
                Cur_img=np.transpose(Cur_img)
            Cur_img=(Cur_img-np.mean(Cur_img))/np.std(Cur_img)

            # plt.imshow(Cur_img)
            # plt.show()
            #Cur_img=np.reshape(Cur_img,20*20)
            ret_data.append([Cur_img])
            #print('\t\t\t Cur_img.shape',Cur_img.shape)
            ret_targets.append(self.Cur_train_category)
            self.Cur_train_category=(1+self.Cur_train_category)%len(categories)
            self.Cur_train_idx[self.Cur_train_category]+=1
            if self.Cur_train_idx[self.Cur_train_category]>=len(self.train_data[self.Cur_train_category]):
                self.Cur_train_idx[self.Cur_train_category]=0
                np.random.shuffle(self.train_data[self.Cur_train_category])

        #print('\t\t\t (np.array(ret_data)).shape',(np.array(ret_data)).shape)
        #print('\t\t\t (torch.from_numpy(np.array(ret_data))).shape',(torch.from_numpy(np.array(ret_data))).shape)
           

        return (torch.from_numpy(np.array(ret_data)),torch.tensor(ret_targets))
        

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class nnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(20*20, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, len(categories))
        )
        self.network2=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 10 x 10

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 5 x 5

            nn.Flatten(), 
            nn.Linear(128*5*5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, len(categories))
        )
        
    def forward(self, input_batch):
        # print(input_batch.shape)
        # exit()
        return self.network2(input_batch)
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc=accuracy(out,labels)
        return (loss,acc)
    
    

print('Constructing Data_Loader() instance')
my_data_loader=Data_Loader()
print('Constructing nnModel() instance')
my_model = nnModel()




def fit(n_epochs,training_batches_per_epoch,training_batch_size, lr, model, data_loader1, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    validation_batches=data_loader1.get_validation_batches_list()
    
    for epoch in range(n_epochs):
        acc_sum1=0
        loss_sum1=0
        for batch_i in range(training_batches_per_epoch):
            batch=data_loader1.train_batch(training_batch_size)
            loss,acc=model.training_step(batch) 
            acc_sum1+=acc
            loss_sum1+=float(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc_sum2=0
        loss_sum2=0
        n_sum2=0
        for batch in validation_batches:
            # predictions=model(valid_inputs)
            # acc=accuracy(predictions,targets)
            loss,acc=model.training_step(batch) 
            loss_sum2+=float(loss)
            acc_sum2+=acc*len(batch)
            n_sum2+=len(batch)

        #print(epoch,float(loss_sum1/training_batches_per_epoch),(100.0*float(acc_sum1/training_batches_per_epoch)),'%',float(loss_sum2/n_sum2),100*float(acc_sum2/n_sum2),'%')
        print("epoch ",epoch, " of ",n_epochs,":")
        print("    tr.  loss: ", float(loss_sum1/training_batches_per_epoch))
        print("    tr.  acc:  ", (100.0*float(acc_sum1/training_batches_per_epoch)),'%')
        print("    val. loss: ", float(loss_sum2/n_sum2))
        print("    val. acc:  ", 100*float(acc_sum2/n_sum2),'%')
        
        # Training Phase 
        # for batch in train_loader:
        #     loss = model.training_step(batch)
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        
        # # Validation phase
        # result = evaluate(model, val_loader)
        # model.epoch_end(epoch, result)
        # history.append(result)

    return history

print('training')
fit(20,10,128,0.001,my_model,my_data_loader) 
torch.save(my_model,'mdl.dat')

print('END')