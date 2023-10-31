import torch
from torchvision.transforms import ToTensor, ToPILImage
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

# Zooms out of a given image, and creates an outpainting mask for the external area.
def create_outpainting_image_and_mask(image, zoom):
    image_tensor = ToTensor()(image).unsqueeze(0)
    _, c, h, w = image_tensor.shape

    center = torch.tensor((h / 2, w / 2)).unsqueeze(0)

    zoom = torch.tensor([zoom, zoom]).unsqueeze(0)
    translate = torch.tensor((0.0, 0.0)).unsqueeze(0)
    angle = torch.tensor([0.0])

    M = get_affine_matrix2d(
        center=center, translations=translate, angle=angle, scale=zoom
    )

    mask_image_tensor = warp_affine(
        image_tensor,
        M=M[:, :2],
        dsize=image_tensor.shape[2:],
        padding_mode="fill",
        fill_value=-1*torch.ones(3),
    )
    mask = torch.where(mask_image_tensor < 0, 1.0, 0.0)

    transformed_image_tensor = warp_affine(
        image_tensor,
        M=M[:, :2],
        dsize=image_tensor.shape[2:],
        padding_mode="border"
    )

    output_mask = ToPILImage()(mask[0])
    output_image = ToPILImage()(transformed_image_tensor[0])

    return output_image, output_mask