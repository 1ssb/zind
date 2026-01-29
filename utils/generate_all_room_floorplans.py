#!/usr/bin/env python3
"""
Batch process entire ZInD dataset to generate per-room binary floorplans.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

# Import the extractor class
from create_per_room_floorplans import PerRoomFloorPlanExtractor


def process_scene(scene_path: Path, args):
    """Process a single scene directory."""
    zind_json = scene_path / "zind_data.json"
    
    if not zind_json.exists():
        return None, f"Missing zind_data.json"
    
    # Create floorplans directory
    floorplans_dir = scene_path / "floorplans"
    floorplans_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize extractor
        extractor = PerRoomFloorPlanExtractor(
            str(zind_json),
            target_size=args.image_size,
            meters_per_image=args.meters_per_image
        )
        
        # Get all complete rooms
        complete_rooms = extractor.get_complete_rooms()
        
        # Metadata for this scene
        scene_metadata = {}
        rooms_processed = 0
        
        # Process rooms
        for floor_id, rooms in complete_rooms.items():
            # Extract floor number
            floor_num = floor_id.split('_')[-1].lstrip('0') or '0'
            
            for complete_room_id in rooms:
                # Extract room geometry
                room_geometry = extractor.get_room_geometry(
                    floor_id,
                    complete_room_id,
                    args.layout_type
                )
                
                if len(room_geometry['vertices']) == 0:
                    continue
                
                # Create filename (sanitize room label for filesystem)
                room_label = room_geometry['label'].replace(' ', '_').replace('/', '_').lower()
                # Remove any other problematic characters
                room_label = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in room_label)
                room_id = complete_room_id.split('_')[-1]
                filename = f"{floor_num}_{room_label}_{room_id}.png"
                output_path = floorplans_dir / filename
                
                # Render binary image
                img, metadata = extractor.render_room_binary(
                    room_geometry,
                    floor_id,
                    str(output_path)
                )
                
                # Store metadata
                scene_metadata[filename] = {
                    'floor_id': floor_id,
                    'floor_number': int(floor_num),
                    'complete_room_id': complete_room_id,
                    'room_label': room_geometry['label'],
                    'num_panos': len(room_geometry['panos']),
                    'num_vertices': len(room_geometry['vertices']),
                    **metadata
                }
                
                rooms_processed += 1
        
        # Save metadata JSON
        metadata_path = floorplans_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(scene_metadata, f, indent=2)
        
        return rooms_processed, None
        
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-room binary floorplans for entire ZInD dataset"
    )
    parser.add_argument(
        "-d", "--data-dir",
        default="data",
        help="Root data directory containing scene folders (default: data)"
    )
    parser.add_argument(
        "--layout-type",
        choices=["layout_raw", "layout_complete", "layout_visible"],
        default="layout_complete",
        help="Type of layout to extract (default: layout_complete)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Output image size in pixels (default: 512)"
    )
    parser.add_argument(
        "--meters-per-image",
        type=float,
        default=10.0,
        help="Physical size to fit in image in meters (default: 10)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start processing from scene index (default: 0)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End processing at scene index (default: process all)"
    )
    parser.add_argument(
        "--scene-id",
        help="Process specific scene ID (e.g., 0000)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # Get all scene directories
    if args.scene_id:
        scene_dirs = [data_dir / args.scene_id]
        if not scene_dirs[0].exists():
            print(f"Error: Scene {args.scene_id} does not exist")
            return
    else:
        scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    # Apply start/end index filtering
    if not args.scene_id:
        if args.start_idx > 0:
            scene_dirs = scene_dirs[args.start_idx:]
        if args.end_idx is not None:
            scene_dirs = scene_dirs[:args.end_idx - args.start_idx]
    
    print(f"ZInD Dataset Per-Room Floorplan Generation")
    print(f"=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Layout type: {args.layout_type}")
    print(f"Image size: {args.image_size}x{args.image_size} pixels")
    print(f"Scale: {args.meters_per_image}m = {args.image_size}px")
    print(f"Pixels per meter: {args.image_size/args.meters_per_image:.2f}")
    print(f"Scenes to process: {len(scene_dirs)}")
    print(f"=" * 60)
    
    # Statistics
    total_rooms = 0
    total_scenes_success = 0
    total_scenes_failed = 0
    failed_scenes = []
    
    # Process each scene
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        
        rooms_count, error = process_scene(scene_dir, args)
        
        if error:
            total_scenes_failed += 1
            failed_scenes.append((scene_id, error))
            tqdm.write(f"✗ {scene_id}: FAILED - {error}")
        else:
            total_scenes_success += 1
            total_rooms += rooms_count
            tqdm.write(f"✓ {scene_id}: {rooms_count} rooms")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Processing Complete")
    print(f"{'=' * 60}")
    print(f"Scenes processed successfully: {total_scenes_success}")
    print(f"Total rooms generated: {total_rooms}")
    print(f"Scenes failed: {total_scenes_failed}")
    
    if failed_scenes:
        print(f"\nFailed scenes:")
        for scene_id, error in failed_scenes[:10]:  # Show first 10
            print(f"  {scene_id}: {error}")
        if len(failed_scenes) > 10:
            print(f"  ... and {len(failed_scenes) - 10} more")
    
    print(f"\nFloorplans saved to: data/<scene_id>/floorplans/")


if __name__ == "__main__":
    main()
