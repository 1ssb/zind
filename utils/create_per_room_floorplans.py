#!/usr/bin/env python3
"""
Create per-room floorplans from ZInD dataset.

The ZInD dataset provides three types of layout annotations per panorama:
1. layout_raw: The raw annotated room geometry from a single panorama
2. layout_complete: The merged/complete room geometry (includes all partial room observations)
3. layout_visible: Only the geometry visible from the current panorama viewpoint

This script demonstrates how to extract per-room floorplans using these annotations.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from PIL import Image, ImageDraw

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

from floor_plan import FloorPlan
from transformations import Transformation2D
from utils import Polygon, PolygonType


class PerRoomFloorPlanExtractor:
    """Extract per-room floorplans from ZInD data."""
    
    def __init__(self, zind_json_path: str, target_size: int = 512, meters_per_image: float = 10.0):
        """
        Initialize the extractor with a ZInD JSON file.
        
        Args:
            zind_json_path: Path to zind_data.json file
            target_size: Size of output image in pixels (default: 512)
            meters_per_image: Physical size to fit in the image in meters (default: 10)
        """
        self.zind_json_path = Path(zind_json_path)
        with open(zind_json_path) as f:
            self.data = json.load(f)
        
        # Parse using the provided FloorPlan class
        self.floor_plan = FloorPlan(zind_json_path)
        
        # Image rendering parameters
        self.target_size = target_size
        self.meters_per_image = meters_per_image
        self.pixels_per_meter = target_size / meters_per_image  # 25.6 for 512/20
        
    def get_complete_rooms(self) -> Dict[str, Dict]:
        """
        Get all complete rooms organized by floor.
        
        Returns:
            Dictionary mapping floor_id -> {complete_room_id: room_data}
        """
        complete_rooms = {}
        merger = self.data['merger']
        
        for floor_id, floor_data in merger.items():
            complete_rooms[floor_id] = {}
            for complete_room_id, complete_room_data in floor_data.items():
                complete_rooms[floor_id][complete_room_id] = complete_room_data
        
        return complete_rooms
    
    def get_room_geometry(
        self, 
        floor_id: str, 
        complete_room_id: str, 
        layout_type: str = "layout_complete"
    ) -> Dict[str, any]:
        """
        Extract geometry for a specific complete room.
        
        Args:
            floor_id: Floor identifier (e.g., "floor_01")
            complete_room_id: Complete room identifier (e.g., "complete_room_01")
            layout_type: One of "layout_raw", "layout_complete", or "layout_visible"
        
        Returns:
            Dictionary containing:
            - vertices: List of 2D vertices in global floor plan coordinates
            - windows: List of window boundaries
            - doors: List of door boundaries
            - openings: List of opening boundaries
            - label: Room label
            - panos: List of panoramas associated with this room
        """
        merger = self.data['merger']
        complete_room_data = merger[floor_id][complete_room_id]
        
        room_info = {
            'vertices': [],
            'windows': [],
            'doors': [],
            'openings': [],
            'internal': [],
            'label': None,
            'panos': []
        }
        
        # For complete layout, we only need to process one panorama
        # since the complete layout is the same for all partial rooms
        processed_complete = False
        
        for partial_room_id, partial_room_data in complete_room_data.items():
            for pano_id, pano_data in partial_room_data.items():
                pano_info = {
                    'pano_id': pano_id,
                    'partial_room_id': partial_room_id,
                    'is_primary': pano_data['is_primary'],
                    'label': pano_data['label'],
                    'image_path': pano_data['image_path']
                }
                room_info['panos'].append(pano_info)
                
                # Set room label from first pano
                if room_info['label'] is None:
                    room_info['label'] = pano_data['label']
                
                # Skip if layout type not available for this pano
                if layout_type not in pano_data:
                    continue
                
                # For complete layout, only process once
                if layout_type == "layout_complete" and processed_complete:
                    continue
                
                # Get the transformation from local to global coordinates
                transformation = Transformation2D.from_zind_data(
                    pano_data["floor_plan_transformation"]
                )
                
                # Transform vertices to global coordinates
                layout = pano_data[layout_type]
                vertices_local = np.array(layout['vertices'])
                vertices_global = transformation.to_global(vertices_local)
                
                if layout_type == "layout_complete":
                    # For complete layout, store the vertices
                    room_info['vertices'] = vertices_global.tolist()
                    
                    # Process internal vertices (e.g., islands)
                    if 'internal' in layout:
                        for internal_vertices in layout['internal']:
                            internal_local = np.array(internal_vertices)
                            internal_global = transformation.to_global(internal_local)
                            room_info['internal'].append(internal_global.tolist())
                    
                    # Process windows, doors, openings
                    for wdo_type in ['windows', 'doors', 'openings']:
                        wdo_data = layout.get(wdo_type, [])
                        if len(wdo_data) > 0:
                            wdo_array = np.array(wdo_data)
                            # WDO are stored as triplets: [left_x, right_x, height]
                            # We only need the left and right x coordinates
                            num_wdo = len(wdo_data) // 3
                            wdo_boundaries = []
                            for i in range(num_wdo):
                                left_right = wdo_data[i*3:i*3+2]
                                wdo_boundaries.extend(left_right)
                            wdo_boundaries = np.array(wdo_boundaries).reshape(-1, 2)
                            wdo_global = transformation.to_global(wdo_boundaries)
                            room_info[wdo_type] = wdo_global.tolist()
                    
                    processed_complete = True
                
                elif layout_type == "layout_visible":
                    # For visible layout, we want per-pano views
                    # Store separately for each pano
                    pano_info['vertices'] = vertices_global.tolist()
                    
                    # Process WDO for visible layout
                    for wdo_type in ['windows', 'doors', 'openings']:
                        wdo_data = layout.get(wdo_type, [])
                        if len(wdo_data) > 0:
                            wdo_array = np.array(wdo_data)
                            num_wdo = len(wdo_data) // 3
                            wdo_boundaries = []
                            for i in range(num_wdo):
                                left_right = wdo_data[i*3:i*3+2]
                                wdo_boundaries.extend(left_right)
                            wdo_boundaries = np.array(wdo_boundaries).reshape(-1, 2)
                            wdo_global = transformation.to_global(wdo_boundaries)
                            pano_info[wdo_type] = wdo_global.tolist()
                
                elif layout_type == "layout_raw":
                    # For raw layout, append from primary panos only
                    if pano_data['is_primary']:
                        if len(room_info['vertices']) == 0:
                            room_info['vertices'] = vertices_global.tolist()
                        
                        # Process WDO for raw layout
                        for wdo_type in ['windows', 'doors', 'openings']:
                            wdo_data = layout.get(wdo_type, [])
                            if len(wdo_data) > 0:
                                wdo_array = np.array(wdo_data)
                                num_wdo = len(wdo_data) // 3
                                wdo_boundaries = []
                                for i in range(num_wdo):
                                    left_right = wdo_data[i*3:i*3+2]
                                    wdo_boundaries.extend(left_right)
                                wdo_boundaries = np.array(wdo_boundaries).reshape(-1, 2)
                                wdo_global = transformation.to_global(wdo_boundaries)
                                if len(room_info[wdo_type]) == 0:
                                    room_info[wdo_type] = wdo_global.tolist()
        
        return room_info
    
    def get_scale_factor(self, floor_id: str) -> Optional[float]:
        """
        Get the scale factor (meters per coordinate unit) for a floor.
        
        Args:
            floor_id: Floor identifier
            
        Returns:
            Scale factor in meters per coordinate unit, or None if not available
        """
        if 'scale_meters_per_coordinate' in self.data:
            return self.data['scale_meters_per_coordinate'].get(floor_id)
        return None
    
    def render_room_binary(
        self,
        room_geometry: Dict,
        floor_id: str,
        output_path: Optional[str] = None
    ) -> Tuple[Image.Image, Dict]:
        """
        Render a room as a binary image (black room on white background).
        
        Args:
            room_geometry: Room geometry dictionary from get_room_geometry()
            floor_id: Floor identifier for scale lookup
            output_path: Path to save the image (optional)
            
        Returns:
            Tuple of (PIL Image, metadata dict with scale info)
        """
        if len(room_geometry['vertices']) == 0:
            return None, {}
        
        # Get scale factor
        scale_meters_per_coord = self.get_scale_factor(floor_id)
        
        # Convert vertices to numpy array
        vertices = np.array(room_geometry['vertices'])
        
        # Calculate bounding box
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        room_size_coords = max_coords - min_coords
        
        # Convert to meters if scale is available
        if scale_meters_per_coord is not None:
            room_size_meters = room_size_coords * scale_meters_per_coord
        else:
            # Assume coordinates are already in meters
            room_size_meters = room_size_coords
            scale_meters_per_coord = 1.0
        
        # Calculate actual pixels per meter based on room size
        max_room_dimension = max(room_size_meters[0], room_size_meters[1])
        
        # Scale to fit within target image while maintaining aspect ratio
        if max_room_dimension > self.meters_per_image:
            # Room is larger than target - scale down
            scale_factor = self.meters_per_image / max_room_dimension
        else:
            # Room is smaller - use standard scale
            scale_factor = 1.0
        
        actual_pixels_per_meter = self.pixels_per_meter * scale_factor
        
        # Create white background image
        img = Image.new('L', (self.target_size, self.target_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Transform vertices to image coordinates
        # Center the room in the image
        center_offset = (self.target_size / 2, self.target_size / 2)
        room_center_coords = (min_coords + max_coords) / 2
        
        def coord_to_pixel(coord):
            """Convert room coordinate to pixel coordinate."""
            # Translate to origin
            translated = coord - room_center_coords
            # Scale to meters then to pixels
            scaled = translated * scale_meters_per_coord * actual_pixels_per_meter
            # Flip Y axis (image Y is top-down) and translate to center
            pixel_x = scaled[0] + center_offset[0]
            pixel_y = center_offset[1] - scaled[1]  # Flip Y
            return (pixel_x, pixel_y)
        
        # Draw main room polygon (black outline only, white fill)
        pixel_vertices = [coord_to_pixel(v) for v in vertices]
        # Draw with a reasonable wall thickness (e.g., 3 pixels)
        wall_thickness = 3
        
        # First fill with white (floor)
        draw.polygon(pixel_vertices, fill=255)
        
        # Then draw black outline (walls)
        # Close the polygon by adding first vertex at the end
        closed_vertices = pixel_vertices + [pixel_vertices[0]]
        draw.line(closed_vertices, fill=0, width=wall_thickness)
        
        # Draw internal polygons if any (black walls)
        for internal in room_geometry.get('internal', []):
            internal_array = np.array(internal)
            pixel_internal = [coord_to_pixel(v) for v in internal_array]
            # Fill with white
            draw.polygon(pixel_internal, fill=255)
            # Draw black outline
            closed_internal = pixel_internal + [pixel_internal[0]]
            draw.line(closed_internal, fill=0, width=wall_thickness)
        
        # Save metadata
        metadata = {
            'pixels_per_meter': float(actual_pixels_per_meter),
            'meters_per_coordinate': float(scale_meters_per_coord),
            'room_size_meters': room_size_meters.tolist(),
            'room_size_coords': room_size_coords.tolist(),
            'image_size_pixels': self.target_size,
            'target_meters_per_image': self.meters_per_image,
            'scale_factor_applied': float(scale_factor)
        }
        
        if output_path:
            img.save(output_path)
            print(f"Saved: {output_path} (scale: {actual_pixels_per_meter:.2f} px/m)")
        
        return img, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-room floorplans as binary images from ZInD dataset"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to zind_data.json file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="per_room_floorplans",
        help="Output directory for images"
    )
    parser.add_argument(
        "--layout-type",
        choices=["layout_raw", "layout_complete", "layout_visible"],
        default="layout_complete",
        help="Type of layout to extract (default: layout_complete)"
    )
    parser.add_argument(
        "--floor",
        help="Specific floor to process (e.g., floor_01)"
    )
    parser.add_argument(
        "--room",
        help="Specific room to process (e.g., complete_room_01)"
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize extractor
    print(f"Loading ZInD data from {args.input}")
    print(f"Image size: {args.image_size}x{args.image_size} pixels")
    print(f"Target scale: {args.meters_per_image}m = {args.image_size}px")
    print(f"Pixels per meter: {args.image_size/args.meters_per_image:.2f}")
    
    extractor = PerRoomFloorPlanExtractor(
        args.input, 
        target_size=args.image_size,
        meters_per_image=args.meters_per_image
    )
    
    # Get all complete rooms
    complete_rooms = extractor.get_complete_rooms()
    
    print(f"\nLayout type: {args.layout_type}")
    
    # Metadata for all rooms
    all_metadata = {}
    
    # Process rooms
    for floor_id, rooms in complete_rooms.items():
        if args.floor and floor_id != args.floor:
            continue
        
        # Extract floor number from floor_id (e.g., "floor_01" -> "1")
        floor_num = floor_id.split('_')[-1].lstrip('0') or '0'
        
        print(f"\n{floor_id} (Floor {floor_num}): {len(rooms)} complete rooms")
        
        for complete_room_id in rooms:
            if args.room and complete_room_id != args.room:
                continue
            
            # Extract room geometry
            room_geometry = extractor.get_room_geometry(
                floor_id, 
                complete_room_id, 
                args.layout_type
            )
            
            if len(room_geometry['vertices']) == 0:
                print(f"  {complete_room_id}: {room_geometry['label']} - SKIPPED (no vertices)")
                continue
            
            # Create clean room label (replace spaces with underscores, lowercase)
            room_label = room_geometry['label'].replace(' ', '_').lower()
            room_id = complete_room_id.split('_')[-1]
            
            # Better filename: floor_number_roomlabel_roomid.png
            filename = f"{floor_num}_{room_label}_{room_id}.png"
            output_path = output_dir / filename
            
            # Render binary image
            img, metadata = extractor.render_room_binary(
                room_geometry, 
                floor_id,
                str(output_path)
            )
            
            # Store metadata
            all_metadata[filename] = {
                'floor_id': floor_id,
                'floor_number': int(floor_num),
                'complete_room_id': complete_room_id,
                'room_label': room_geometry['label'],
                'num_panos': len(room_geometry['panos']),
                'num_vertices': len(room_geometry['vertices']),
                **metadata
            }
            
            print(f"  {complete_room_id}: {room_geometry['label']}")
            print(f"    Saved as: {filename}")
    
    # Save metadata JSON
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nProcessed {len(all_metadata)} rooms")
    print(f"Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
