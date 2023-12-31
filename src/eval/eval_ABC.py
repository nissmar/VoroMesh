import numpy as np
import joblib
import trimesh
from sklearn.neighbors import KDTree
from tqdm import tqdm
import argparse

sample_num = 100000
all_models = "src/eval/abc_ordered.txt"
nw_list = 'src/eval/not_watertight_ABC_test.txt'
f1_threshold = 0.003


def get_cd_f1_nc(name):
    idx = name[0]
    gt_obj_name = name[1]
    pred_obj_name = name[2]

    # load gt
    gt_mesh = trimesh.load(gt_obj_name)

    gt_points, gt_indexs = gt_mesh.sample(sample_num, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_indexs]
    # load pred
    pred_mesh = trimesh.load(pred_obj_name)
    try:
        pred_points, pred_indexs = pred_mesh.sample(
            sample_num, return_index=True)
        pred_normals = pred_mesh.face_normals[pred_indexs]
    except:
        print('\n\n\n WARNING \n\n\n')
        pred_points = np.zeros((1, 3))
        pred_normals = np.zeros((1, 3))

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

    # write_ply_point("temp/"+str(idx)+"_gt.ply", gt_edge_points)
    # write_ply_point("temp/"+str(idx)+"_pred.ply", pred_edge_points)

    # ecd ef1

    # print(idx, cd1, cd2, nc, f1)
    return idx, cd1, cd2, f1, nc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation on ABC dataset')
    parser.add_argument('pred_dir', type=str, help='Path to .obj shapes')
    parser.add_argument('-gt_dir', type=str, default="data/ABC/val/obj/",
                        help='Path to .obj groundtruth shapes')

    args = parser.parse_args()

    fin = open(nw_list, 'r')
    nw = [name.strip()[:-5] for name in fin.readlines()]
    fin.close()

    fin = open(all_models, 'r')
    obj_names = [name.strip()[:-5] for name in fin.readlines()]
    obj_names_old = obj_names[int(len(obj_names)*0.8):]
    obj_names = [e for e in obj_names_old if not (e in nw)]
    fin.close()

    obj_names_len = len(obj_names)

    # prepare list of names
    list_of_list_of_names = []
    true_idx = 0
    for idx in range(len(obj_names_old)):
        if not obj_names_old[idx] in nw:
            gt_obj_name = "{}/{}.obj".format(args.gt_dir, obj_names_old[idx])
            pred_obj_name = "{}/test_{}.obj".format(args.pred_dir, idx)
            list_of_list_of_names.append(
                [idx, gt_obj_name, pred_obj_name])
            true_idx += 1

    out = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_cd_f1_nc)
                                     (name) for name in tqdm(list_of_list_of_names))
    out = np.array(out)
    mean_scores = out.mean(0)
    print('CD (x 1e-5): {:.3f}  &  F1: {:.3f}  &  NC: {:.3f}'.format(
        mean_scores[2]*1e5, mean_scores[3], mean_scores[4]))
