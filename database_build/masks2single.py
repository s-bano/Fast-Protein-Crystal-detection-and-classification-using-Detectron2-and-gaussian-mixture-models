import numpy as np
from PIL import Image
from pathlib import Path

def combine_instance_masks_to_binary_mask(input_root, output_dir):
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for instance_dir in sorted(input_root.iterdir()):
        if not instance_dir.is_dir():
            continue

        binary_mask = None
        count = 0

        for mask_file in sorted(instance_dir.glob("mask_*.png")):
            mask = np.array(Image.open(mask_file))
            mask = (mask > 0).astype(np.uint8)

            if binary_mask is None:
                binary_mask = np.zeros(mask.shape, dtype=np.uint8)

            binary_mask = np.maximum(binary_mask, mask)
            count += 1

        if count == 0:
            print(f"⚠ No masks found in {instance_dir.name}")
            continue

        output_path = output_dir / f"{instance_dir.name}_mask.png"
        Image.fromarray(binary_mask * 255).save(output_path)
        print(f"✔ {output_path.name} (combined {count} masks)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py input_folder output_folder")
    else:
        combine_instance_masks_to_binary_mask(sys.argv[1], sys.argv[2])