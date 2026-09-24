import torch
import segmentation_models_pytorch as smp


CLASS_NAMES = ("floor", "wall", "door", "window")


def build_model(weights_path: str):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model
