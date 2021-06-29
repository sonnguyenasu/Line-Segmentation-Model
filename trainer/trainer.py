from torch import nn
from loss import Criterion
from torch import optim
import torch
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def IoU(y_pred, masks, threshold=0.6):
    '''compute mean IoU exaclty using prediction and target'''
    y_pred, y_true = (y_pred >= threshold).float(), torch.round(masks)
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

class Trainer(nn.Module):
    def __init__(self, args, model, data_loader, test_loader):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.iteration = 0  # which iteration is currently running
        self.args = args
        self.device = args.device
        self.writer = SummaryWriter(log_dir='tensorboard_output/')
        if args.resume is not None:
            self.load()
        self.model = self.model.to(self.device)
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        self.criterion = Criterion()
    def train(self):
        for epoch in range(self.args.epoch):
            self.train_step()

    def train_step(self):
        running_loss = 0
        for i, data in enumerate(self.data_loader):
            self.iteration = self.iteration + 1
            independent, dependent = data
            independent, dependent = independent.to(
                self.device), dependent.to(self.device)

            prediction = self.model(independent)
            loss = self.criterion(dependent, prediction)
            running_loss += loss
            if i % self.args.log_frequency == 1:
                print("Iteration:",self.iteration,end=', ')
                print('loss:', loss.item())
                img_grid = make_grid(independent, nrow=2)
                mask_grid= make_grid(dependent, nrow=2)
                pred_grid= make_grid(prediction, nrow=2)
                
                self.writer.add_image('input image',img_grid)
                self.writer.add_image('mask',mask_grid)
                self.writer.add_image('prediction',pred_grid)
                self.writer.add_scalar('training loss', running_loss/self.args.log_frequency, self.iteration)
                running_loss = 0
                
            if self.iteration % self.args.save_frequency == 1:
                # iou = self.eval()
                # if iou > self.best:
                #     self.save('best.pth')
                #     self.best = iou
                self.save()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        pass

    def eval(self):
        self.model.eval()
        iou = 0
        if self.test_loader is None:
            return 0
        for i, data in enumerate(tqdm(self.test_loader)):
            img, mask = data
            img, mask = img.to(self.device), mask.to(self.device)
            prediction = self.model(img)
            iou += IoU(prediction[:,:1,:,:], mask[:,:1,:,:])
        self.model.train()
        return(iou/i)

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(
            self.args.save_path, '{:06d}.pth'.format(self.iteration)))

    def load(self):
        self.model.load_state_dict(torch.load(self.args.resume))
