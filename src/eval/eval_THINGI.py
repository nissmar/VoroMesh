import numpy as np
import joblib
import trimesh
from sklearn.neighbors import KDTree
from tqdm import tqdm
import argparse


sample_num = 100000
all_models = "src/eval/thingi32_names.txt"
f1_threshold = 0.003


def get_cd_f1_nc(name, rescale, sample_num, f1_threshold):
    idx = name[0]
    gt_obj_name = name[1]
    pred_obj_name = name[2]

    # load gt
    gt_mesh = trimesh.load(gt_obj_name)
    gt_points, gt_indexs = gt_mesh.sample(sample_num, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_indexs]
    # load pred
    pred_mesh = trimesh.load(pred_obj_name)
    pred_mesh.vertices[:] /= rescale

    pred_points, pred_indexs = pred_mesh.sample(
        sample_num, return_index=True)
    pred_normals = pred_mesh.face_normals[pred_indexs]

    # cd and nc and f1

    # from gt to pred
    pred_tree = KDTree(pred_points)
    dist, inds = pred_tree.query(gt_points, k=1)
    recall = np.sum(dist < f1_threshold) / float(len(dist))
    gt2pred_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    gt2pred_mean_cd2 = np.mean(dist)
    neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(gt_normals*neighbor_normals, axis=1))
    gt2pred_nc = np.mean(dotproduct)

    # from pred to gt
    gt_tree = KDTree(gt_points)
    dist, inds = gt_tree.query(pred_points, k=1)
    precision = np.sum(dist < f1_threshold) / float(len(dist))
    pred2gt_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    pred2gt_mean_cd2 = np.mean(dist)
    neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(pred_normals*neighbor_normals, axis=1))
    pred2gt_nc = np.mean(dotproduct)

    cd1 = gt2pred_mean_cd1+pred2gt_mean_cd1
    cd2 = gt2pred_mean_cd2+pred2gt_mean_cd2
    nc = (gt2pred_nc+pred2gt_nc)/2
    if recall+precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0
    return idx, cd1, cd2, f1, nc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluation on Thingi32 dataset')
    parser.add_argument('pred_dir', type=str, help='Path to .obj shapes')

    parser.add_argument('-formatted', type=bool,
                        default=False, help='for name formatting')
    parser.add_argument('-gt_dir', type=str, default="data/Thingi32/obj/",
                        help='Path to .obj groundtruth shapes')
    args = parser.parse_args()

    fin = open(all_models, 'r')
    obj_names = [name.strip() for name in fin.readlines()]
    fin.close()

    obj_names_len = len(obj_names)

    # prepare list of names
    list_of_list_of_names = []
    for idx in range(len(obj_names)):
        if not (obj_names[idx] in ["96481", "58168"]):
            gt_obj_name = args.gt_dir + obj_names[idx] + ".obj"
            if args.formatted:
                pred_obj_name = "{}/test_{}.obj".format(args.pred_dir, idx)
            else:
                pred_obj_name = "{}/{}.obj".format(
                    args.pred_dir, obj_names[idx])
            list_of_list_of_names.append(
                [idx, gt_obj_name, pred_obj_name])

    rescale = 1 if args.formatted else 2
    out = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_cd_f1_nc)
                                     (name, rescale, sample_num, f1_threshold) for name in tqdm(list_of_list_of_names))
    out = np.array(out)

    mean_scores = out.mean(0)
    print(args.pred_dir)
    print('CD (x 1e-5): {:.3f}  &  F1: {:.3f}  &  NC: {:.3f}'.format(
        mean_scores[2]*1e5, mean_scores[3], mean_scores[4]))
