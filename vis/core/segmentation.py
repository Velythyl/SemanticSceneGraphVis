from __future__ import annotations

from typing import Any

import open3d as o3d


class SparseSegmentMasker:
    def split_objects(
        self,
        sparse_pcd: o3d.geometry.PointCloud | None,
        segments_anno: dict[str, Any] | None,
    ) -> list[o3d.geometry.PointCloud]:
        if sparse_pcd is None:
            return []

        if not segments_anno:
            return [sparse_pcd]

        seg_groups = segments_anno.get("segGroups", [])
        if not seg_groups:
            return [sparse_pcd]

        split_objects: list[o3d.geometry.PointCloud] = []
        for ann in seg_groups:
            if not isinstance(ann, dict):
                continue

            indices = ann.get("segments", [])
            if not isinstance(indices, list) or not indices:
                continue

            obj = sparse_pcd.select_by_index(indices)
            if not obj.is_empty():
                split_objects.append(obj)

        if not split_objects:
            return [sparse_pcd]

        return split_objects
