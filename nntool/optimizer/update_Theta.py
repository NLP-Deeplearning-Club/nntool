

def update_Theta(Theta:'matrix',alpha:'float',delta:"matrix"):
    """牛顿法,alpha是步长,delta是梯度"""
    return theta-(alpha * delta)
