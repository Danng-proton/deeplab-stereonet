def color_to_gray(img):
    img = img.permute(0, 2, 3, 1)
    img_gray = 0.299*img[:,:,:,0]+0.587*img[:,:,:,1]+0.114*img[:,:,:,2]
    img_gray = img_gray.unsqueeze(3).permute(0, 3, 1, 2)
    return img_gray

def x_gradient_1order(img):
    img = img.permute(0,2,3,1)
    img_l = img[:,:,1:,:] - img[:,:,:-1,:]
    img_r = img[:,:,-1,:] - img[:,:,-2,:]
    img_r = img_r.unsqueeze(2)
    img  = torch.cat([img_l, img_r], 2).permute(0, 3, 1, 2)
    return img

def y_gradient_1order(img):
    # pdb.set_trace()
    img = img.permute(0,2,3,1)
    img_u = img[:,1:,:,:] - img[:,:-1,:,:]
    img_d = img[:,-1,:,:] - img[:,-2,:,:]
    img_d = img_d.unsqueeze(1)
    img  = torch.cat([img_u, img_d], 1).permute(0, 3, 1, 2)
    return img

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    print("r.shape")
    printf(r.shape)
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad