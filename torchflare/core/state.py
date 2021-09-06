class State(dict):
    """Container object exposing keys as attributes.

    State objects extend dictionaries by enabling values to be accessed by key,
    `state["value_key"]`, or by an attribute, `state.value_key`.

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        """Setattr method.

        Args:
            key: The input key.
            value: The corresponding value to the key.
        """
        self[key] = value

    def __dir__(self):
        """Method to return all the keys."""
        return self.keys()

    def __getattr__(self, key):
        """Method to access value associated with the key.

        Args:
            key: The input key.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


__all__ = ["State"]
