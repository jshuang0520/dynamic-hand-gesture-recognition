import torch

def get_mock_dataloader(batch_size, num_classes):
    """Generates synthetic video tensors. Replaces circular dependency issues."""
    # Simulates: [Batch, Frames, Channels, Height, Width]
    data = torch.randn(4 * batch_size, 16, 3, 224, 224)
    targets = torch.randint(0, num_classes, (4 * batch_size,))
    dataset = torch.utils.data.TensorDataset(data, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)