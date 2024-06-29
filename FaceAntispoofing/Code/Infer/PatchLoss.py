import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)
    

class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m_l=0.4, m_s=0.1):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l]
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input: 
            x shape (N, in_features)
            labels shape (N)
        '''
        
        # assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        # Reshape x to have shape (N, in_features)
        x = x.view(x.size(0), -1)
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)
        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        m = torch.tensor([self.m[ele] for ele in labels]).to(x.device)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        L = numerator - torch.log(denominator)
        
        return - torch.mean(L)



class PatchLoss(nn.Module):

    def __init__(self, alpha1=1.0, alpha2=1.0):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sim_loss = SimilarityLoss()
        self.amsm_loss = AdMSoftmaxLoss(384, 2)

    
    def forward(self, x1, x2, label):
        amsm_loss1 = self.amsm_loss(x1, label.type(torch.long).squeeze())
        amsm_loss2 = self.amsm_loss(x2, label.type(torch.long).squeeze())
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim_loss = self.sim_loss(x1, x2)
        loss = self.alpha1 * sim_loss + self.alpha2 * (amsm_loss1 + amsm_loss2)
        
        return loss