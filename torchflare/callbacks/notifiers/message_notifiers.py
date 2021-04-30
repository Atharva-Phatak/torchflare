"""Implements notifiers for slack and discord."""
import json
from abc import ABC

import requests

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


def prepare_data(logs: dict):
    """Function to prepare the data according to the type of message.

    Args:
        logs: Dictionary containing the metrics and loss values.

    Returns:
        string in the same format as logs.
    """
    val = [f"{key} : {value}" for key, value in logs.items()]
    text = "\n".join(val)
    return text


class SlackNotifierCallback(Callbacks, ABC):
    """Class to Dispatch Training progress to your Slack channel."""

    def __init__(self, webhook_url: str):
        """Constructor method for SlackNotifierCallback.

        Args:
            webhook_url : Slack webhook url
        """
        super(SlackNotifierCallback, self).__init__(order=CallbackOrder.EXTERNAL)
        self.webhook_url = webhook_url

    def epoch_end(self):
        """This function will dispatch messages to your Slack channel.

        Raises:
            ValueError: If connection to slack channel could not be established.
        """
        data = {"text": prepare_data(self.exp.exp_logs)}

        response = requests.post(self.webhook_url, json.dumps(data), headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            raise ValueError(
                "Request to slack returned an error {}, the response is:\n{}".format(
                    response.status_code, response.text
                )
            )


class DiscordNotifierCallback(Callbacks, ABC):
    """Class to Dispatch Training progress to your Discord Sever."""

    def __init__(self, exp_name: str, webhook_url: str):
        """Constructor method for DiscordNotifierCallback.

        Args:
            exp_name : The name of your experiment bot. (Can be anything)
            webhook_url : The webhook url of your discord server/channel.
        """
        super(DiscordNotifierCallback, self).__init__(order=CallbackOrder.EXTERNAL)
        self.exp_name = exp_name
        self.webhook_url = webhook_url

    def epoch_end(self):
        """This function will dispatch messages to your discord server/channel."""
        data = {
            "username": self.exp_name,
            "embeds": [{"description": prepare_data(self.exp.exp_logs)}],
        }
        response = requests.post(self.webhook_url, json.dumps(data), headers={"Content-Type": "application/json"})

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)
