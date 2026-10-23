from cs336_basics.optimizer import SGD
import torch

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
lrs = [1, 1e1, 1e2, 1e3]
for lr in lrs:
    print("lr: ", lr)
    opt = SGD([weights], lr=lr)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.