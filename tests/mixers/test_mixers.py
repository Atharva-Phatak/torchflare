from torchflare.batch_mixers.mixers import cutmix, mixup, get_collate_fn
import torch


x = torch.randn(4, 3, 256, 256)
targets = torch.tensor([0, 1, 0, 1])

ds = torch.utils.data.TensorDataset(x, targets)


def test_mixup():
    dl = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    op, y = mixup(batch=batch, alpha=0.35)

    assert torch.is_tensor(op) is True
    assert isinstance(y, (tuple, list)) is True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) is True
    assert torch.is_tensor(targets_b) is True
    assert isinstance(lam, (int, float)) is True


def test_cutmix():
    dl = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    op, y = cutmix(batch=batch, alpha=0.35)

    assert torch.is_tensor(op) is True
    assert isinstance(y, (tuple, list)) is True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) is True
    assert torch.is_tensor(targets_b) is True
    assert isinstance(lam, (int, float)) is True


def test_collate_fn_mixup():

    mixup_collate_fn = get_collate_fn(mixer_name="mixup", alpha=0.35)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=mixup_collate_fn)
    op, y = next(iter(dl))

    assert torch.is_tensor(op) is True
    assert isinstance(y, (tuple, list)) is True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) is True
    assert torch.is_tensor(targets_b) is True
    assert isinstance(lam, (int, float)) is True


def test_collate_fn_cutmix():
    mixup_collate_fn = get_collate_fn(mixer_name="cutmix", alpha=0.35)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=mixup_collate_fn)
    op, y = next(iter(dl))

    assert torch.is_tensor(op) is True
    assert isinstance(y, (tuple, list)) is True

    targets_a, targets_b, lam = y
    assert torch.is_tensor(targets_a) is True
    assert torch.is_tensor(targets_b) is True
    assert isinstance(lam, (int, float)) is True
