
def acc(pred, labels):
    _, pred_inds = pred.max(dim=1)
    pred_inds = pred_inds.view(-1)
    labels = labels.view(-1)
    acc = pred_inds.eq(labels).float().mean()
    return acc