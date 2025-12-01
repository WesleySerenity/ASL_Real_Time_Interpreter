import os
import shutil

DATA_DIR = "data"
BAD_ROOT = "bad_images"

os.makedirs(BAD_ROOT, exist_ok=True)

for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    bad_dir = os.path.join(class_dir, "_bad")
    if not os.path.isdir(bad_dir):
        continue

    print(f"Processing bad dir: {bad_dir}")

    # Make a per-class folder under bad_images for organization
    target_class_dir = os.path.join(BAD_ROOT, class_name)
    os.makedirs(target_class_dir, exist_ok=True)

    for fname in os.listdir(bad_dir):
        src = os.path.join(bad_dir, fname)
        if not os.path.isfile(src):
            continue

        base, ext = os.path.splitext(fname)
        dst = os.path.join(target_class_dir, fname)

        # Avoid overwriting duplicates
        counter = 1
        while os.path.exists(dst):
            dst = os.path.join(target_class_dir, f"{base}_{counter}{ext}")
            counter += 1

        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

    # Remove now-empty _bad directory
    try:
        os.rmdir(bad_dir)
    except OSError:
        pass  # not empty for some reason, just skip
