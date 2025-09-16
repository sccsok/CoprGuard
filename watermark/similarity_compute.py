import cv2
import glob
import numpy as np


def cosine_similarity(img1, img2):
    """Compute cosine similarity between two images after flattening."""
    img1 = img1.astype(np.float32).reshape(-1)
    img2 = img2.astype(np.float32).reshape(-1)
    norm1 = np.linalg.norm(img1)
    norm2 = np.linalg.norm(img2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(img1, img2) / (norm1 * norm2)

def compare_single_image(source_path, target_path):
    """Compare similarity between two images (with flip consideration)."""
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))

    flipped_target = cv2.flip(target_img, 1)
    cc = max(
        cosine_similarity(source_img, target_img),
        cosine_similarity(source_img, flipped_target)
    )
    print(f"Similarity between {source_path} and {target_path}: {cc:.4f}")
    return cc

def compare_folder_images(folder_path, target_path, resize=(128, 128), threshold=None):
    """Compare all images in a folder with a target image and return similarities."""
    target_img = cv2.imread(target_path)
    target_img = cv2.resize(target_img, resize)
    flipped_target = cv2.flip(target_img, 1)

    image_paths = glob.glob(f"{folder_path}/*.png")
    image_paths.sort()

    results = []
    for img_path in image_paths:
        source_img = cv2.imread(img_path)
        source_img = cv2.resize(source_img, resize)

        cc = max(
            cosine_similarity(source_img, target_img),
            cosine_similarity(source_img, flipped_target)
        )
        results.append((img_path, cc))

    results.sort(key=lambda x: x[1], reverse=True)
    for path, score in results:
        print(f"{path}: {score:.4f}")

    if threshold is not None:
        binarized = [1 if score >= threshold else 0 for _, score in results]
        ratio = sum(binarized) / len(binarized)
        print(f"Images >= {threshold:.5f}: {sum(binarized)} / {len(binarized)} ({ratio:.2%})")

    return results


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Compare images or folders with watermark")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "folder"], 
        required=True,
        help="Choose single image comparison or folder comparison"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None, 
        help="Path to the image (for single mode)"
    )
    parser.add_argument(
        "--watermark", 
        type=str, 
        required=True, 
        help="Path to watermark image"
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        default=None, 
        help="Path to image folder (for folder mode)"
    )
    parser.add_argument(
        "--resize", 
        type=int, 
        nargs=2, 
        default=(128, 128), 
        help="Resize size (h, w)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.928, 
        help="Threshold for comparison"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "single":
        if args.image is None:
            raise ValueError("You must provide --image for single mode")
        compare_single_image(args.image, args.watermark)

    elif args.mode == "folder":
        if args.folder is None:
            raise ValueError("You must provide --folder for folder mode")
        compare_folder_images(
            args.folder,
            args.watermark,
            resize=tuple(args.resize),
            threshold=args.threshold
        )