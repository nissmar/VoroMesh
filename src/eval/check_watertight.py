import os
import igl
import trimesh
import argparse


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Test watertightness of the output shapes.')
    parser.add_argument('voromesh_dir', type=str,
                        help='path to input voromeshes')
    return parser


def evaluate(src_dir):
    wrong_geometry = 0
    wrong_topology = 0
    empty = 0
    watertight = 0
    num_models = 0
    for model_name in os.listdir(src_dir):
        if (".obj" in model_name or ".off" in model_name) and not (model_name[0] == '.' or '96481' in model_name or '58168' in model_name):
            num_models += 1
            v, f = igl.read_triangle_mesh(src_dir + model_name)
            if len(v) > 0:
                wt = not trimesh.Trimesh(v, f, process=False).is_watertight
                wg = int(
                    os.popen("src/cpp_utils/build/self_intersect {}".format(src_dir+model_name)).read()[:-1])
                if wt:
                    wrong_topology += 1
                if wg > 0:
                    wrong_geometry += 1
                if not (wt) and not (wg):
                    watertight += 1
            else:
                empty += 1
    return (wrong_geometry, wrong_topology, empty, watertight / num_models)


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    scores = evaluate(args.voromesh_dir+'/')
    print(args.voromesh_dir)
    print('Percentage of watertight meshes: {}%'.format(scores[3]*100))
    print('Self-intersecting models: {}'.format(scores[0]))
    print('Non-closed models: {}'.format(scores[1]))
    print('Empty models: {}'.format(scores[2]))
