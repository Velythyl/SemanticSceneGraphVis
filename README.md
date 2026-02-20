# vis

Visualize semantic scene graphs and annotate them by placing 3D bounding boxes, interactively!

<iframe width="560" height="315" src="https://youtu.be/bkwQ_Hzc6Sk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



[<img src="https://youtu.be/bkwQ_Hzc6Sk" width="600" height="300"
/>](https://youtu.be/bkwQ_Hzc6Sk)



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