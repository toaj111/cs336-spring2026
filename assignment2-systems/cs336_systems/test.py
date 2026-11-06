import torch

if __name__ == "__main__":
    A = torch.randn(6, 4)
    print(A)
    A = A.max(dim=-1)
    print(A)
    A = A.values
    print(A, A.shape)