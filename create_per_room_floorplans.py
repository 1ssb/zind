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
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add code directory to path
sys.path.append(str(Path(__file__).parent / "code"))

from floor_plan import FloorPlan
from transformations import Transformation2D
from utils import Polygon, PolygonType


class PerRoomFloorPlanExtractor:
    """Extract per-room floorplans from ZInD data."""
    
    def __init__(self, zind_json_path: str):
        """
        Initialize the extractor with a ZInD JSON file.
        
        Args:
            zind_json_path: Path to zind_data.json file
        """
        self.zind_json_path = Path(zind_json_path)
        with open(zind_json_path) as f:
            self.data = json.load(f)
        
        # Parse using the provided FloorPlan class
        self.floor_plan = FloorPlan(zind_json_path)
        
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
    
    def visualize_room(
        self, 
        room_geometry: Dict,
        output_path: Optional[str] = None,
        show_wdo: bool = True
    ):
        """
        Visualize a single room's floorplan.
        
        Args:
            room_geometry: Room geometry dictionary from get_room_geometry()
            output_path: Path to save the visualization (optional)
            show_wdo: Whether to show windows/doors/openings
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot room vertices
        if len(room_geometry['vertices']) > 0:
            vertices = np.array(room_geometry['vertices'])
            # Close the polygon
            vertices_closed = np.vstack([vertices, vertices[0]])
            ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], 
                   'k-', linewidth=2, label='Room boundary')
            ax.fill(vertices_closed[:, 0], vertices_closed[:, 1], 
                   alpha=0.1, color='gray')
        
        # Plot internal polygons (e.g., islands)
        for internal in room_geometry.get('internal', []):
            internal_array = np.array(internal)
            internal_closed = np.vstack([internal_array, internal_array[0]])
            ax.plot(internal_closed[:, 0], internal_closed[:, 1], 
                   'k-', linewidth=2)
            ax.fill(internal_closed[:, 0], internal_closed[:, 1], 
                   alpha=0.2, color='gray')
        
        if show_wdo:
            # Plot windows
            windows = room_geometry.get('windows', [])
            if len(windows) > 0:
                windows_array = np.array(windows)
                for i in range(0, len(windows_array), 2):
                    ax.plot([windows_array[i, 0], windows_array[i+1, 0]], 
                           [windows_array[i, 1], windows_array[i+1, 1]], 
                           'b-', linewidth=4, label='Window' if i == 0 else '')
            
            # Plot doors
            doors = room_geometry.get('doors', [])
            if len(doors) > 0:
                doors_array = np.array(doors)
                for i in range(0, len(doors_array), 2):
                    ax.plot([doors_array[i, 0], doors_array[i+1, 0]], 
                           [doors_array[i, 1], doors_array[i+1, 1]], 
                           'g-', linewidth=4, label='Door' if i == 0 else '')
            
            # Plot openings
            openings = room_geometry.get('openings', [])
            if len(openings) > 0:
                openings_array = np.array(openings)
                for i in range(0, len(openings_array), 2):
                    ax.plot([openings_array[i, 0], openings_array[i+1, 0]], 
                           [openings_array[i, 1], openings_array[i+1, 1]], 
                           'r--', linewidth=4, label='Opening' if i == 0 else '')
        
        # Plot panorama positions
        for pano in room_geometry['panos']:
            if 'vertices' in pano:
                # For visible layouts, we can show the pano-specific view
                pass
            # Could add camera positions here if needed
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Room: {room_geometry['label']}")
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and visualize per-room floorplans from ZInD dataset"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to zind_data.json file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="per_room_floorplans",
        help="Output directory for visualizations"
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize extractor
    print(f"Loading ZInD data from {args.input}")
    extractor = PerRoomFloorPlanExtractor(args.input)
    
    # Get all complete rooms
    complete_rooms = extractor.get_complete_rooms()
    
    print(f"\nLayout type: {args.layout_type}")
    print(f"Available layouts per panorama:")
    print(f"  - layout_raw: Raw annotated room from a single panorama")
    print(f"  - layout_complete: Complete merged room (all partial observations)")
    print(f"  - layout_visible: Only geometry visible from panorama viewpoint")
    
    # Process rooms
    for floor_id, rooms in complete_rooms.items():
        if args.floor and floor_id != args.floor:
            continue
        
        print(f"\n{floor_id}: {len(rooms)} complete rooms")
        
        for complete_room_id in rooms:
            if args.room and complete_room_id != args.room:
                continue
            
            # Extract room geometry
            room_geometry = extractor.get_room_geometry(
                floor_id, 
                complete_room_id, 
                args.layout_type
            )
            
            print(f"  {complete_room_id}: {room_geometry['label']}")
            print(f"    Panoramas: {len(room_geometry['panos'])}")
            if len(room_geometry['vertices']) > 0:
                print(f"    Vertices: {len(room_geometry['vertices'])}")
            
            # Create visualization
            output_path = output_dir / f"{floor_id}_{complete_room_id}_{args.layout_type}.png"
            extractor.visualize_room(room_geometry, str(output_path))
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
