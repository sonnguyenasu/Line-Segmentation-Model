from torch import nn
from loss import Criterion
from torch import optim
import torch
import os


class Trainer(nn.Module):
    def __init__(self, args, model, data_loader):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.iteration = 0  # which iteration is currently running
        self.args = args
        self.device = args.device
        if args.resume is not None:
            self.load()
            self.iteration = int(args.resume.split('/')[-1].split('.')[0])
        self.model = self.model.to(self.device)
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)

    def train(self):
        for epoch in range(self.args.epoch):
            self.train_step()

    def train_step(self):
        criterion = Criterion()
        for i, data in enumerate(self.data_loader):
            self.iteration = self.iteration + 1
            independent, dependent = data
            independent, dependent = independent.to(
                self.device), dependent.to(self.device)

            prediction = self.model(independent)
            loss = criterion(dependent, prediction)

            if i % self.args.log_frequency == 0:
                print("Iteration:",self.iteration,end=', ')
                print('loss:', loss.item())
                import cv2
                import numpy as np
                pred = np.array(
                    (prediction[0, :, :, :]).detach().cpu().permute(1, 2, 0))
                # pred = np.where(pred > 0.3, 1, 0)
                cv2.imwrite('pred.jpg', np.exp(pred)*255)
                cv2.imwrite('gt.jpg', np.array(
                    (dependent[0, :, :, :]*255).detach().cpu().permute(1, 2, 0), dtype='uint8'))
                
                in_img = np.array(
                    (independent[0, :, :, :]*255).detach().cpu().permute(1, 2, 0), dtype='uint8')
                cv2.imwrite('in.jpg', in_img[:,:,::-1])
            if self.iteration % self.args.save_frequency == 1:
                # self.scheduler.step(val_score)
                # print('Saving...')
                self.save()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        pass

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(
            self.args.save_path, '{:06d}.pth'.format(self.iteration)))

    def load(self):
        self.model.load_state_dict(torch.load(self.args.resume))
