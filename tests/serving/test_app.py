from torchflare.serving import GradioApp, API_TYPES
import pytest
import torch
from dataclasses import asdict
import unittest.mock as mock

model = torch.nn.Linear(10, 1)
post_process_fn = lambda x: torch.sigmoid(x)
pre_process_fn = lambda x: torch.tensor(x)


def test_gradio_app():
    api_types = asdict(API_TYPES())
    for api in api_types.values():
        app = GradioApp(app_title="App", description="Dummy app", model=model, api_type=api)
        assert app


@pytest.mark.parametrize("pre_process_func, post_process_func", [(None, None), (pre_process_fn, post_process_fn)])
def test_forward_pass(pre_process_func, post_process_func):

    app = GradioApp(
        app_title="App",
        description="Dummy app",
        model=model,
        api_type=API_TYPES.image_classification,
        post_process_func=post_process_fn,
        pre_process_func=pre_process_fn,
    )
    x = torch.randn(10)
    op = app.forward_pass(x)
    assert torch.is_tensor(op) is True


@mock.patch("torchflare.serving.gradio_app.gradio")
def test_app(mock_gradio):
    mock_gradio.Interface = mock.MagicMock()
    app = GradioApp(
        app_title="App",
        description="Dummy app",
        model=model,
        api_type=API_TYPES.image_classification,
        post_process_func=post_process_fn,
        pre_process_func=pre_process_fn,
    )
    app.run(interface_config=None)

    mock_gradio.Interface.assert_called_with(
        fn=app.forward_pass,
        inputs=app.gradio_input_type,
        outputs=app.gradio_output_type,
        title=app.app_title,
        description=app.description,)
