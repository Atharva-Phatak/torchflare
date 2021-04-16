import torch
import torchflare.modules as modules

# To-do: Add more handcrafted tests.


def check_layer(layer):
    embedding = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(10)

    output = layer(embedding, target)
    assert output.shape == (3, 10)

    output = layer(embedding)
    assert output.shape == (3, 10)


def check_se_blocks(block, in_channels):

    x = torch.randn(2, in_channels, 8, 8)
    op = block(x)

    assert x.shape == op.shape


def test_arcface():
    check_layer(modules.ArcFace(5, 10, s=1.31, m=0.5))


def test_cosface():
    check_layer(modules.CosFace(5, 10, s=1.31, m=0.5))


def test_airface():
    check_layer(modules.LiArcFace(5, 10, s=1.31, m=0.5))


def test_amsoftmax():
    check_layer(modules.AMSoftmax(5, 10, s=1.31, m=0.5))


def test_se_blocks():
    in_channels = 64
    check_se_blocks(modules.CSE(in_channels=in_channels), in_channels=in_channels)
    check_se_blocks(modules.SSE(in_channels=in_channels), in_channels=in_channels)
    check_se_blocks(modules.SCSE(in_channels=in_channels, r=8), in_channels=in_channels)
