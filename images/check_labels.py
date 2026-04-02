"""Check which images have fire/smoke labels vs backgrounds."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"D:\USA\Sems\4_Spring 2026 Sem\AI for Engineers\Multimodal\lokesh\images\images"

# Check first 30 images
train_imgs = sorted(os.listdir(os.path.join(base, "data", "train", "images")))[:30]
print("First 30 training images (sorted alphabetically):")
for img in train_imgs:
    lbl = os.path.splitext(img)[0] + ".txt"
    lbl_path = os.path.join(base, "data", "train", "labels", lbl)
    if os.path.exists(lbl_path):
        content = open(lbl_path).read().strip()
        if content:
            classes = set(line.split()[0] for line in content.split("\n") if line.strip())
            labels = []
            if "0" in classes: labels.append("FIRE")
            if "1" in classes: labels.append("HUMAN")
            if "2" in classes: labels.append("SMOKE")
            print(f"  {img}: {' + '.join(labels)}")
        else:
            print(f"  {img}: BACKGROUND")
    else:
        print(f"  {img}: NO LABEL")

# Find some images WITH fire/smoke for testing
print("\nLooking for images with fire/smoke labels...")
fire_images = []
for f in sorted(os.listdir(os.path.join(base, "data", "train", "labels"))):
    content = open(os.path.join(base, "data", "train", "labels", f)).read().strip()
    if content and ("0 " in content or content.startswith("0")):
        img_name = os.path.splitext(f)[0] + ".jpg"
        fire_images.append(img_name)
    if len(fire_images) >= 20:
        break
print(f"Found {len(fire_images)} fire images: {fire_images[:10]}")  