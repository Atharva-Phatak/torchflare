# flake8: noqa
import os

from torchflare.callbacks.notifiers import DiscordNotifierCallback, SlackNotifierCallback
import pytest


@pytest.mark.skip(reason="Requires secret webhook url.")
def test_slack():

    slack_n = SlackNotifierCallback(webhook_url=os.environ.get("SLACK_WEBHOOK"))

    acc = 10
    f1 = 10
    loss = 100
    for epoch in range(10):

        d = {"acc": acc, "f1": f1, "loss": loss}
        acc += 10
        f1 += 10

        loss = loss / 10

        slack_n.epoch_end()

    print("Slack testing completed")


@pytest.mark.skip(reason="Requires secret webhook url.")
def test_discord():

    discord_n = DiscordNotifierCallback(webhook_url=os.environ.get("DISCORD_WEBHOOK"), exp_name="Test_discord")

    acc = 10
    f1 = 10
    loss = 100
    for epoch in range(10):
        d = {"acc": acc, "f1": f1, "loss": loss}
        acc += 10
        f1 += 10

        loss = loss / 10

        discord_n.epoch_end()

    print("Discord Testing Completed")
