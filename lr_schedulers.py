def reset_scheduler(epoch, lr):
    if epoch == 100:
        return 1.0
    else:
        return lr