def euclidean_loss(output, target):
    batch_size = output.shape[0]
    return (output - target).abs().sum() / batch_size