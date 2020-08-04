'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to build a neural network.
'''
import os
import platform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataloader import HairNetDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoder
        # self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(3, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        # decoder
        self.fc1 = nn.Linear(1*1*512, 4*4*256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        # MLP
        # Position
        self.branch1_fc1 = nn.Linear(256*32*32, 32)
        self.branch1_fc2 = nn.Linear(32, 32)
        self.branch1_fc3 = nn.Linear(32, 32*32*300)
        # Curvature
        self.branch2_fc1 = nn.Linear(256*32*32, 32)
        self.branch2_fc2 = nn.Linear(32, 32)
        self.branch2_fc3 = nn.Linear(32, 32*32*100)
        
    def forward(self, x):
        # encoder
        # x = F.relu(self.conv1(x)) # (batch_size, 32, 128, 128)
        x = F.relu(self.conv2(x)) # (batch_size, 64, 64, 64)
        x = F.relu(self.conv3(x)) # (batch_size, 128, 32, 32)
        x = F.relu(self.conv4(x)) # (batch_size, 256, 16, 16)
        x = F.relu(self.conv5(x)) # (batch_size, 512, 8, 8)
        x = F.max_pool2d(x, 8) # (batch_size, 512, 1, 1)
        # decoder
        x = x.view(-1, 1*1*512)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.conv6(x)) # (batch_size, 256, 4, 4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # (batch_size, 256, 8, 8)
        x = F.relu(self.conv7(x)) # (batch_size, 256, 8, 8)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners = False) # (batch_size, 256, 16, 16)
        x = F.relu(self.conv8(x)) # (batch_size, 256, 16, 16)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # (batch_size, 256, 32, 32)
        x = x.view(-1, 256*32*32)
        # MLP
        # Position
        branch1_x = F.relu(self.branch1_fc1(x))
        branch1_x = F.relu(self.branch1_fc2(branch1_x))
        branch1_x = F.relu(self.branch1_fc3(branch1_x))
        branch1_x = branch1_x.view(-1, 100, 3, 32, 32)
        # Curvature
        branch2_x = F.relu(self.branch2_fc1(x))
        branch2_x = F.relu(self.branch2_fc2(branch2_x))
        branch2_x = F.relu(self.branch2_fc3(branch2_x))
        branch2_x = branch2_x.view(-1, 100, 1, 32, 32)
        x = [branch1_x, branch2_x]
        return torch.cat(x, 2) # (batch_size, 100, 4, 32, 32)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, output, convdata, visweight):
        # removing nested for-loops (0.238449s -> 0.001860s)
        pos_loss = visweight[:,:,:,:].reshape(1,-1).mm(torch.pow((convdata[:,:,0:3,:,:]-output[:,:,0:3,:,:]),2).reshape(-1, 3)).sum()
        cur_loss = visweight[:,:,:,:].reshape(1,-1).mm(torch.pow((convdata[:,:,3,:,:]-output[:,:,3,:,:]),2).reshape(-1, 1)).sum()
        # print(pos_loss/(convdata.shape[0]*convdata.shape[1]*1024.0) + cur_loss/(convdata.shape[0]*convdata.shape[1]*1024.0))
        return pos_loss/(convdata.shape[0]*convdata.shape[1]*1024.0) + cur_loss/(convdata.shape[0]*convdata.shape[1]*1024.0)


class MyPosEvaluation(nn.Module):
    def __init__(self):
        super(MyPosEvaluation, self).__init__()
    def forward(self, output, convdata):
        # removing nested for-loops (0.083651s -> 0.001371s)
        loss = torch.mean(torch.abs(convdata[:,:,0:3,:,:]-output[:,:,0:3,:,:]))
        return loss


class MyCurEvaluation(nn.Module):
    def __init__(self):
        super(MyCurEvaluation, self).__init__()
    def forward(self, output, convdata):
        # removing nested for-loops (0.080810s -> 0.000990s)
        loss = torch.mean(torch.abs(convdata[:,:,3,:,:]-output[:,:,3,:,:]))
        return loss

class PosMSE(nn.Module):
    def __init__(self):
        super(PosMSE, self).__init__()
    def forward(self, output, convdata, visweight, verbose = False):
        visweight = visweight[:,:,:,:].reshape(1,-1)
        e_squared = torch.pow((convdata[:,:,0:3,:,:]-output[:,:,0:3,:,:]),2).reshape(-1, 3)
        loss = visweight.mm(e_squared).sum()
        
        if verbose:
            print('[Position Loss]')

            visweight1 = (visweight == 10).float()*10
            vis_loss = visweight1.mm(e_squared).sum()
            print("\tvis: ", (vis_loss/(convdata.shape[0]*convdata.shape[1]*1024.0)).item())
            
            visweight2 = (visweight == 0.1).float()*0.1
            inv_loss = visweight2.mm(e_squared).sum()
            print("\tinv: ", (inv_loss/(convdata.shape[0]*convdata.shape[1]*1024.0)).item())

        return loss/(convdata.shape[0]*convdata.shape[1]*1024.0)

class CurMSE(nn.Module):
    def __init__(self):
        super(CurMSE, self).__init__()
    def forward(self, output, convdata, visweight, verbose = False):
        visweight = visweight[:,:,:,:].reshape(1,-1)
        e_squared = torch.pow((convdata[:,:,3,:,:]-output[:,:,3,:,:]),2).reshape(-1, 1)
        loss = visweight.mm(e_squared).sum()
        
        if verbose:
            print('[Curvature Loss]')

            visweight1 = (visweight == 10).float()*10
            vis_loss = visweight1.mm(e_squared).sum()
            print("\tvis: ", (vis_loss/(convdata.shape[0]*convdata.shape[1]*1024.0)).item())
            
            visweight2 = (visweight == 0.1).float()*0.1
            inv_loss = visweight2.mm(e_squared).sum()
            print("\tinv: ", (inv_loss/(convdata.shape[0]*convdata.shape[1]*1024.0)).item())

        return loss/(convdata.shape[0]*convdata.shape[1]*1024.0)

def train(root_dir, load_epoch = None):
    print('This is the programme of training.')

    log_path = root_dir+'/log.txt'
    loss_pic_path = root_dir+'/loss.png'
    weight_save_path = root_dir + '/weight/'
    debug_weight_save_path = root_dir + '/debug/'
    debug_log_path = root_dir+'/debug/log.txt'
    debug_loss_pic_path = root_dir+'/debug/loss.png'

    # build model
    print('Initializing Network...')
    net = Net()
    net.cuda()
    loss = MyLoss()
    loss.cuda()
    # load weight if possible
    start_epoch = 0
    if load_epoch != None:
        weight_load_path = 'weight/{}_weight.pt'.format(load_epoch) 
        net.load_state_dict(torch.load(weight_load_path))
        start_epoch = int(load_epoch)
        print("start epoch:", start_epoch+1)
    # set hyperparameter
    EPOCH = 500
    BATCH_SIZE = 300
    LR = 1e-4
    # set parameter of log
    PRINT_STEP = 10 # batch
    LOG_STEP = 100 # batch
    WEIGHT_STEP = 1 # epoch
    LR_STEP = 250 # change learning rate
    # load data
    print('Setting Dataset and DataLoader...')
    train_data = HairNetDataset(project_dir=root_dir,train_flag=1,noise_flag=1)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    # set optimizer    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_list = []
    print('Training...')
    
    if torch.cuda.device_count() >= 1:
        print("gpu count:", torch.cuda.device_count())
        net = nn.DataParallel(net)

    #niet = Net()
    #net.load_state_dict(torch.load(save_path))
    
    for i in range(start_epoch, EPOCH):
        epoch_loss = 0.0
        # change learning rate
        if (i+1)%LR_STEP == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                param_group['lr'] = current_lr * 0.5
        for j, data in enumerate(train_loader, 0):
            img, convdata, visweight = data
            img = img.cuda()
            convdata = convdata.cuda()
            visweight = visweight.cuda()
            # img (batch_size, 3, 128, 128)     
            # convdata (batch_size, 100, 4, 32, 32)
            # visweight (batch_size, 100, 32, 32)

            # zero the parameter gradients
            optimizer.zero_grad()
            output = net(img) #img (batch_size, 100, 4, 32, 32)
            my_loss = loss(output, convdata, visweight)

            # debug 
            if i == 0 and j == 0:
                if not os.path.exists(debug_log_path):
                    with open(debug_log_path, 'w') as f:
                        f.write('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(my_loss.item()) + '\n')
                        print('Debug of writing log.txt!')
                else:
                    with open(debug_log_path, 'a') as f:
                        f.write('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(my_loss.item()) + '\n')        
                        print('Debug of writing log.txt!')
                save_path = debug_weight_save_path + 'weight.pt'

                torch.save(net.state_dict(), save_path)
                print('Debug of saving model!')
                debug_loss_list = []
                debug_loss_list.append(my_loss.item())
                plt.plot(debug_loss_list)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.xlim(0, EPOCH-1)
                plt.savefig(debug_loss_pic_path)
                print('Debug of drawing loss picture!')

            epoch_loss += my_loss.item()
            my_loss.backward()
            optimizer.step()
            if (j+1)%PRINT_STEP == 0:
                print('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(my_loss.item()))
            if (j+1)%LOG_STEP == 0:
                if not os.path.exists(log_path):
                    with open(log_path, 'w') as f:
                        f.write('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(my_loss.item()) + '\n')    
                else:
                    with open(log_path, 'a') as f:
                        f.write('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(my_loss.item()) + '\n')        
        if (i+1)%WEIGHT_STEP == 0:
            new_state_dict = OrderedDict()
            for k, v in net.state_dict().items():
                name = k[7:]
                new_state_dict[name]=v
            save_path = weight_save_path + str(i+1).zfill(6) + '_weight_v2.pt'
            torch.save(new_state_dict, save_path)
        loss_list.append(epoch_loss)
    print('Finish...')
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, EPOCH-1)
    plt.savefig(loss_pic_path)


def test(root_dir, weight_path):
    print('This is the programme of testing.')
    BATCH_SIZE = 32
    # load model
    print('Building Network...')
    net = Net()
    net.cuda()
    pos_error = PosMSE()
    pos_error.cuda()
    cur_error = CurMSE()
    cur_error.cuda()
    print('Loading Network...')
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    print('Loading Dataset...')
    test_data = HairNetDataset(project_dir=root_dir,train_flag=0,noise_flag=0)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    # load testing data
    print('Testing...')
    for i, data in enumerate(test_loader, 0):
        img, convdata, visweight = data
        img = img.cuda()
        convdata = convdata.cuda()
        visweight = visweight.cuda()
        output = net(img)
        pos = pos_error(output, convdata, visweight, verbose = True)
        cur = cur_error(output, convdata, visweight)
        print(str(BATCH_SIZE*(i+1)) + '/' + str(len(test_data)) + ', Position loss: ' + str(pos.item()) + ', Curvature loss: ' + str(cur.item()))


def demo(root_dir, weight_path):
    print('This is the programme of demo.')
    BATCH_SIZE = 1
    # load model
    print('Building Network...')
    net = Net()
    net.cuda()
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    test_data = HairNetDataset(project_dir=root_dir,train_flag=0,noise_flag=0)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # load testing data
    for i, data in enumerate(test_loader, 0):
        img, _, _ = data
        
        cv2.imshow('', img[0].numpy().T) # input orientation
        
        img = img.cuda()
        output = net(img)
        
        strands = output[0].cpu().detach().numpy() # hair strands
        with open ('demo/demo.convdata', 'wb') as wf:
            np.save(wf, strands)
        print(np.swapaxes(strands[:,:3,:,:],0,1).shape)
        hair_pos = np.swapaxes(np.swapaxes(strands[:,:3,:,:],0,1).reshape(3,-1), 0,1)
        print(hair_pos.shape)
        with open ('demo/demo.txt', 'w') as wf:
            np.savetxt(wf, hair_pos)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

