from typing import Callable, Optional

import gradio
import torch

from torchflare.serving.api_type import API_TYPES


class GradioApp:
    """Gradio App for various Tasks."""

    def __init__(
        self,
        app_title: str,
        description: str,
        api_type: str,
        model: torch.nn.Module,
        preprocess_func: Callable = None,
        postprocess_func: Callable = None,
        gradio_output_type=None,
        **kwargs,
    ):

        self.app_title = app_title
        self.description = description
        self.api_type = api_type
        self.model = model
        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func
        self.gradio_output_type = gradio_output_type if gradio_output_type is not None else "json"
        self.gradio_input_type = self._create_gradio_inputs()

    def _create_gradio_inputs(self):
        if self.api_type in [API_TYPES.image_classification, API_TYPES.object_detection]:
            return gradio.inputs.Image(label="Input")
        elif self.api_type == API_TYPES.text_classification:
            return gradio.inputs.Textbox(lines=4, label="Input")
        else:
            raise NotImplementedError(f"{self.api_type} is not implemented yet.")

    def forward_pass(self, x):
        """Forward pass for model including preprocessing \
        and post-processing."""
        if self.preprocess_func is not None:
            x = self.preprocess_func(x)

        with torch.no_grad():
            op = self.model(x)

        if self.postprocess_func is not None:
            op = self.postprocess_func(op)

        return op

    def create_interface(self, interface_config: Optional[dict] = None):
        """Method to create gradio interface."""
        if not interface_config:
            interface_config = {}

        return gradio.Interface(
            fn=self.forward_pass,
            inputs=self.gradio_input_type,
            outputs=self.gradio_output_type,
            title=self.app_title,
            description=self.description,
            **interface_config,
        )

    def run(self, interface_config=None, share: bool = False):
        """Method to run gradio app."""
        iface = self.create_interface(interface_config=interface_config)
        iface.launch(share=share)
