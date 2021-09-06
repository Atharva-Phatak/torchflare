from dataclasses import dataclass


@dataclass
class API_TYPES:
    """Available API Types."""

    image_classification: str = "IMAGE-CLASSIFICATION"
    object_detection: str = "OBJECT-DETECTION"
    text_classification: str = "TEXT-CLASSIFICATION"
