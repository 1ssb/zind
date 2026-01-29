#!/usr/bin/env python3
"""
Re-run ONLY Stage 1 (Perceptual Hash) deduplication.
This will give us the exact 9,709 unique images.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    import imagehash
    from PIL import Image
except ImportError:
    print("Error: Required libraries not available")
    sys.exit(1)

DATA_DIR = "/home/group/cvml-datasets/zind/data"
OUTPUT_FILE = "/home/group/cvml-datasets/zind/unique_floorplans_stage1_only.txt"
HASH_SIZE = 16
HASH_THRESHOLD = 8

print("=" * 70)
print("Re-running Stage 1 (Perceptual Hash) Only")
print("=" * 70)
print()

# Load all floorplan paths
print("Loading floorplan paths...")
floorplan_paths = []
data_dir = Path(DATA_DIR)
for floorplan_file in sorted(data_dir.glob("*/floorplans/*.png")):
    floorplan_paths.append(str(floorplan_file.absolute()))

print("Found {} floorplan images".format(len(floorplan_paths)))
print()

# Compute hashes
print("Computing perceptual hashes...")
hash_map = {}
failed = 0

for image_path in tqdm(floorplan_paths, desc="Hashing"):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_hash = imagehash.phash(img, hash_size=HASH_SIZE)
            hash_map[image_path] = str(img_hash)
    except:
        failed += 1

print("Successfully hashed: {}".format(len(hash_map)))
print("Failed: {}".format(failed))
print()

# Group similar hashes
print("Finding duplicates (threshold={})...".format(HASH_THRESHOLD))
duplicate_groups = defaultdict(list)
processed = set()
paths = list(hash_map.keys())

for i, path1 in enumerate(tqdm(paths, desc="Grouping")):
    if path1 in processed:
        continue
    
    hash1 = imagehash.hex_to_hash(hash_map[path1])
    group = [path1]
    
    for path2 in paths[i+1:]:
        if path2 in processed:
            continue
        
        hash2 = imagehash.hex_to_hash(hash_map[path2])
        hamming_dist = hash1 - hash2
        
        if hamming_dist <= HASH_THRESHOLD:
            group.append(path2)
            processed.add(path2)
    
    if len(group) > 1:
        duplicate_groups[path1] = group[1:]
    
    processed.add(path1)

# Calculate unique images
total_duplicates = sum(len(dups) for dups in duplicate_groups.values())
all_duplicates = set()
for dups in duplicate_groups.values():
    all_duplicates.update(dups)

unique_images = [p for p in floorplan_paths if p not in all_duplicates]

print()
print("=" * 70)
print("STAGE 1 RESULTS")
print("=" * 70)
print("Total images:          {}".format(len(floorplan_paths)))
print("Duplicate groups:      {}".format(len(duplicate_groups)))
print("Total duplicates:      {}".format(total_duplicates))
print("Unique images:         {}".format(len(unique_images)))
print("Reduction:             {:.2f}%".format((total_duplicates / len(floorplan_paths)) * 100))
print("=" * 70)
print()

# Write output
print("Writing unique images to {}".format(OUTPUT_FILE))
with open(OUTPUT_FILE, 'w') as f:
    for img_path in sorted(unique_images):
        f.write("{}\n".format(img_path))

print("Done!")
print()
print("Use this file for Stage 1 only deduplication:")
print("  {}".format(OUTPUT_FILE))
