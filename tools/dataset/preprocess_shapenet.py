import argparse
import sys
from pathlib import Path

import bpy

import os

MAX_SIZE = 5e7


_package_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    # Drop blender arguments.
    argv = sys.argv[5:]
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='shapenet_dir', type=Path)
    parser.add_argument(dest='out_dir', type=Path)
    parser.add_argument('--strip-materials', action='store_true')
    parser.add_argument('--out-name', type=str, required=True)
    args = parser.parse_args(args=argv)

    paths = sorted(args.shapenet_dir.glob('**/model_normalized.obj'))

    for i, path in enumerate(paths):
        print(f"*** [{i+1}/{len(paths)}]")

        model_size = os.path.getsize(path)
        if model_size > MAX_SIZE:
            print("Model too big ({} > {})".format(model_size, MAX_SIZE))
            continue

        synset_id = path.parent.parent.parent.name
        model_id = path.parent.parent.name
        # if model_id != '831918158307c1eef4757ae525403621':
        #     continue
        print(f"Processing {path!s}")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.obj(filepath=str(path),
                                 use_edges=True,
                                 use_smooth_groups=True,
                                 use_split_objects=True,
                                 use_split_groups=True,
                                 use_groups_as_vgroups=False,
                                 use_image_search=True)

        if len(bpy.data.objects) > 10:
            print("Too many objects. Skipping for now..")
            continue

        if args.strip_materials:
            print("Deleting materials.")
            for material in bpy.data.materials:
                material.user_clear()
                bpy.data.materials.remove(material)

        for obj_idx, obj in enumerate(bpy.data.objects):
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            print("Clearing split normals and removing doubles.")
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
            bpy.ops.mesh.remove_doubles()
            bpy.ops.mesh.normals_make_consistent(inside=False)

            print("Unchecking auto_smooth")
            obj.data.use_auto_smooth = False

            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            print("Adding edge split modifier.")
            mod = obj.modifiers['EdgeSplit']
            mod.split_angle = 20

            bpy.ops.object.mode_set(mode='OBJECT')

            print("Applying smooth shading.")
            bpy.ops.object.shade_smooth()

            print("Running smart UV project.")
            bpy.ops.uv.smart_project()

            bpy.context.active_object.select_set(state=False)

        out_path = args.out_dir / synset_id / model_id / 'models' / args.out_name
        print(out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        bpy.ops.export_scene.obj(filepath=str(out_path),
                                 group_by_material=True,
                                 keep_vertex_order=True,
                                 use_normals=True, use_uvs=True,
                                 use_materials=True,
                                 check_existing=False)
        print("Saved to {}".format(out_path))


if __name__ == '__main__':
    main()
