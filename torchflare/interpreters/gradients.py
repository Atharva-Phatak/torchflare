"""Save Hooks."""


# flake8: noqa
class SaveHooks:
    """Class to save hooks for activations and gradients."""

    def __init__(self, layer):
        """Constructor method for SaveHooks class."""
        self.activations = None
        self.gradients = None
        self.forward_hook = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_full_backward_hook(self.hook_fn_grad)

    # skipcq: PYL-W0613
    def hook_fn_act(self, module, act_input, op):  # noqa
        """Method to save activation."""
        self.activations = op

    # skipcq: PYL - W0613
    def hook_fn_grad(self, module, grad_input, gradient_output):  # noqa
        """Method to  save gradients."""
        self.gradients = gradient_output[0]

    def remove(self):  # noqa
        """Method to remove hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
