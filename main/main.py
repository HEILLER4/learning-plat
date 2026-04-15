import cv2
import csv
import os


def get_image_info(filepath: str) -> dict:
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")

    height, width, channels = img.shape
    size_bytes = os.path.getsize(filepath)

    return {
        "filename": os.path.basename(filepath),
        "width": width,
        "height": height,
        "channels": channels,
        "size_kb": round(size_bytes / 1024, 2),
    }


def scan_folder(folder: str) -> list[dict]:
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    results = []

    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(supported):
            full_path = os.path.join(folder, fname)
            try:
                info = get_image_info(full_path)
                results.append(info)
            except ValueError as e:
                print(f"Skipping: {e}")

    return results


def write_report(data: list[dict], output_path: str) -> None:
    if not data:
        print("No images found.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    output = sys.argv[2] if len(sys.argv) > 2 else "report.csv"
    data = scan_folder(folder)
    write_report(data, output)
