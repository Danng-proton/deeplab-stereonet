import torch
import numpy as np
import test_model
from torch.autograd import Variable



##x1=torch.randn(5,5,3,1)
##x2=torch.randn(5,5,3,1)
with torch.cuda.device(1):
    x1 = Variable(torch.cuda.FloatTensor(np.random.randn(1,3,10,10)),requires_grad=True)
    x2 = Variable(torch.cuda.FloatTensor(np.random.rand(1,3,10,10)),requires_grad=True)
    print 'x1.is_contiguous()',x1.is_contiguous()
    print 'x2.is_contiguous()',x2.is_contiguous()
    # print x1
    # x1 = torch.cuda.FloatTensor(np.random.randn(1,3,10,10))
    # x2 = torch.cuda.FloatTensor(np.random.randn(1,3,10,10))
    x3 = Variable(torch.cuda.FloatTensor(np.random.randn(1,3,10,8)))
    print 'x3.requires_grad',x3.requires_grad
    print 'x2.is_contiguous()',x3.is_contiguous()
    model=test_model.FlowNetC().cuda()
    # for param in model.parameters():
        # print 'param',param
    # print model
    # x1.requires_grad=True
    # x2.requires_grad=True
    print 'x1.requires_grad',x1.requires_grad
    result=model(x1,x2)
    print 'result.is_contiguous()',result.is_contiguous()
    print 'result.requires_grad',result.requires_grad
    loss=torch.sum(result-x3)
    print 'loss.is_contiguous()',loss.is_contiguous() is True
    # loss.backward()
    # print result.shape