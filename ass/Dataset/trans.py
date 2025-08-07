import os
from PIL import Image


def resize_and_pad(img: Image.Image, target_size: tuple, pad_color=(0, 0, 0)):
    """
    Resize an image while preserving aspect ratio and pad to the target size.
    """
    original_size = img.size  # (width, height)
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

    # Resize image
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

    # Create new image with padding color
    new_img = Image.new(img.mode, target_size, pad_color)
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_img.paste(img_resized, paste_position)

    return new_img


def process_dataset(input_dir, mask_dir, output_img_dir, output_mask_dir, target_size=(512, 512)):
    """
    Process all images and masks in the given directories:
    - Resize + pad each RGB/NRG image and corresponding mask to the target size.
    - Save the results to the output directories.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    masks_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    for img_name, masks_name in zip(image_files,masks_files):
        # Paths
        img_path = os.path.join(input_dir, img_name)
        mask_path = os.path.join(mask_dir, masks_name)  # assumes same filename

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # single channel mask

        # Resize and pad
        img_processed = resize_and_pad(img, target_size, pad_color=(0, 0, 0))
        mask_processed = resize_and_pad(mask, target_size, pad_color=0)

        # Save outputs
        img_processed.save(os.path.join(output_img_dir, img_name.replace('RGB_', '')))
        mask_processed.save(os.path.join(output_mask_dir, img_name.replace('RGB_', '')))

# Example usage:
process_dataset(
    input_dir="./RGB_images",
    mask_dir="./masks",
    output_img_dir="./resize/RGB_images",
    output_mask_dir="./resize/masks",
    target_size=(512, 512)
)
