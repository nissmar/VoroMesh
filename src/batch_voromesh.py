from time import time
from single_voromesh import create_voromesh, DEFAULTS
import os
from tqdm import tqdm
import torch
import utils.mesh_tools as mt

src_dir = './data/Thingi32/obj/'  # Thingi32 is stored
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for grid_n in [32, 64, 128]:
    out_dir = "out/direct_voromesh_{}/".format(grid_n)
    T0 = time()
    opt_time = 0
    extract_time = 0

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for model_name in tqdm(os.listdir(src_dir)):
        v, f, samples = mt.load_and_sample_shape(
            model_name, src_dir, DEFAULTS["samples_fac"]*grid_n**2, rescale_f='NDC')
        V, t0 = create_voromesh(samples, v, f, grid_n, return_time=True)
        opt_time += t0

        # export .pt pytorch file
        # torch.save(
        #     V, 'voronois/batch_voronoi_{}_NDC_norm/voronoi_{}.pt'.format(grid_n, model_name[:-4]))
        t1 = time()
        nv, nf = V.to_mesh()
        extract_time += time()-t1
        mt.export_obj(nv, nf, '{}/{}'.format(out_dir, model_name[:-4]))

    print('TOTAL TIME: {}, OPT_TIME: {}, EXTRACT_TIME: {}'.format(
        time()-T0, opt_time, extract_time))
