from torchflare.batch_mixers.mixers import cutmix , mixup , get_collate_fn
import torch


x = torch.randn(4, 3, 256, 256)
y = torch.tensor([0, 1, 0, 1])

ds = torch.utils.data.TensorDataset(x, y)
dl = torch.utils.data.DataLoader(ds, batch_size=2)


def test_mixup():

    batch = next(iter(dl))
    op, y = mixup(batch=batch, alpha=0.35)

    assert torch.is_tensor(op) == True
    assert isinstance(y, (tuple, list)) == True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) == True
    assert torch.is_tensor(targets_b) == True
    assert isinstance(lam, (int, float)) == True


def test_cutmix():

    batch = next(iter(dl))
    op, y = cutmix(batch=batch, alpha=0.35)

    assert torch.is_tensor(op) == True
    assert isinstance(y, (tuple, list)) == True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) == True
    assert torch.is_tensor(targets_b) == True
    assert isinstance(lam, (int, float)) == True


def test_collate_fn_mixup():

    mixup_collate_fn = get_collate_fn(mixer_name="mixup", alpha=0.35)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=mixup_collate_fn)
    x, y = next(iter(dl))

    assert torch.is_tensor(x) == True
    assert isinstance(y, (tuple, list)) == True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) == True
    assert torch.is_tensor(targets_b) == True
    assert isinstance(lam, (int, float)) == True


def test_collate_fn_cutmix():
    mixup_collate_fn = get_collate_fn(mixer_name="cutmix", alpha=0.35)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=mixup_collate_fn)
    x, y = next(iter(dl))

    assert torch.is_tensor(x) == True
    assert isinstance(y, (tuple, list)) == True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) == True
    assert torch.is_tensor(targets_b) == True
    assert isinstance(lam, (int, float)) == True
