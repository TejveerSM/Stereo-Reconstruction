import torch

def px3_loss(softmax_scores, gt, weights):    
    loss = 0
    for i in range(softmax_scores.size(0)):
        gt_disparity = gt[i]
        pred_scores = softmax_scores[i, gt_disparity.item()-2:gt_disparity.item()+2+1]
        if (pred_scores.size(0) == 0):
        	pred_scores = torch.zeros(5, device='cuda:0')
        	print('exception')
        sl = torch.mul(pred_scores, weights).sum()
        loss -= sl
        
    return loss