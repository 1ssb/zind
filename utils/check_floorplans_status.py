#!/usr/bin/env python3
"""
Check status of floorplan generation across all scenes.
"""

from pathlib import Path
import json

def main():
    data_dir = Path("data")
    
    # Get all scene directories
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    total_scenes = len(scene_dirs)
    scenes_with_floorplans = 0
    scenes_without_floorplans = 0
    total_rooms = 0
    scenes_no_zind_json = 0
    
    missing_scenes = []
    
    print(f"Checking {total_scenes} scenes...")
    print("=" * 60)
    
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        zind_json = scene_dir / "zind_data.json"
        floorplans_dir = scene_dir / "floorplans"
        metadata_file = floorplans_dir / "metadata.json"
        
        if not zind_json.exists():
            scenes_no_zind_json += 1
            continue
        
        if floorplans_dir.exists() and metadata_file.exists():
            scenes_with_floorplans += 1
            
            # Count rooms in this scene
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    num_rooms = len(metadata)
                    total_rooms += num_rooms
            except:
                pass
        else:
            scenes_without_floorplans += 1
            missing_scenes.append(scene_id)
    
    # Results
    print(f"\nResults:")
    print(f"=" * 60)
    print(f"Total scenes: {total_scenes}")
    print(f"Scenes with zind_data.json: {total_scenes - scenes_no_zind_json}")
    print(f"Scenes without zind_data.json: {scenes_no_zind_json}")
    print(f"\nFloorplan Generation Status:")
    print(f"  ✓ Processed: {scenes_with_floorplans}")
    print(f"  ✗ Missing: {scenes_without_floorplans}")
    print(f"  Total rooms generated: {total_rooms}")
    
    if scenes_without_floorplans > 0:
        print(f"\nMissing scenes ({len(missing_scenes)}):")
        for i, scene_id in enumerate(missing_scenes[:20]):
            print(f"  {scene_id}", end="")
            if (i + 1) % 10 == 0:
                print()
            else:
                print(" ", end="")
        if len(missing_scenes) > 20:
            print(f"\n  ... and {len(missing_scenes) - 20} more")
        else:
            print()
    
    # Completion percentage
    if total_scenes > scenes_no_zind_json:
        completion = (scenes_with_floorplans / (total_scenes - scenes_no_zind_json)) * 100
        print(f"\nCompletion: {completion:.1f}%")
    
    print(f"=" * 60)

if __name__ == "__main__":
    main()
