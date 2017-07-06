def L2_Norm(lambd:'>0',n,W):
    result = (lambd/(2*n))*((W**2).sum())
    return result
