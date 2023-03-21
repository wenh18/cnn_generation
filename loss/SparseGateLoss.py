import torch.nn as nn
import torch

class sparse_gate_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bass_criterion=nn.CrossEntropyLoss()
    def forward(self,logists,label,kernel_selection):
        label=label.float()
        #交叉熵部分
        loss = self.bass_criterion(logists, label.long())
        if kernel_selection is not None:
            c=kernel_selection.size()[-1]*kernel_selection.size()[-2]*kernel_selection.size()[-3]
            #限制kernel使用率，这里为0.3,意思是尽可能往30%的利用率去走
            loss=loss+0.1*torch.pow(((torch.norm(kernel_selection,p=1))/c-0.3),2)
        return loss