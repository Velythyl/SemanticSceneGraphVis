# vis

Visualize semantic scene graphs and annotate them by placing 3D bounding boxes, interactively!


<p align="center">
	<a href="https://youtu.be/bkwQ_Hzc6Sk">
		<img src="https://img.youtube.com/vi/bkwQ_Hzc6Sk/maxresdefault.jpg" width="600" height="300" style="filter: grayscale(75%);" />
	</a>
</p>

<p align="center">Click image to watch video</p>



# Installation

```bash
uv sync
```

# Running it

```bash
uv run vis/vis_with_viser.py
```

You can tune paths by changing `vis/conf`

# Annotations

Use the Box Annotator feature to place boxes. Save to file using `Save / Load`

# Supported input maps

- ConceptGraphs https://github.com/sachaMorin/concept-nodes
- A dense point cloud (name it as `dense_point_cloud.pcd`)
- You can display videos by naming them as `rgb.mp4`, `depth.mp4` and `rgbd.mp4`

## Notes

- Libraries used to build this software have their own licenses, which you should be aware of before running this software.