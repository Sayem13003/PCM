# sig_t_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=5.75):
        super(sig_t, self).__init__()
        self.register_parameter(name='A', param=nn.Parameter(torch.ones(num_classes*7, num_classes)))
        self.init = init
        num_classes=20
        self.A.to(device)
        self.identity = torch.eye(num_classes).repeat(7, 1).to(device)

    def forward(self):
        # Clone the tensor to avoid in-place modification
        soft_modified = self.A.clone()
        soft_modified[:, :20] += self.init * self.identity
        T_C = F.softmax(soft_modified, dim=1)
        return T_C
