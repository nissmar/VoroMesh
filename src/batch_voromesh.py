from time import time
from single_voromesh import create_voromesh, DEFAULTS
import os
from tqdm import tqdm
import torch
import utils.mesh_tools as mt

src_dir = '../data/thingy32/groundtruths/'  # Thingi32 is stored
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpp_compiled = os.path.isfile('src/cpp_utils/build/voromesh')
if not (cpp_compiled):
    print('WARNING: CGAL voromesh not found, using scipy mesh extraction with NO WATERTIGHTNESS GUARANTEES. Please compile cpp_utils.')

for grid_n in [32, 64, 128]:
    out_dir = "out/direct_voromesh_{}/".format(grid_n)
    T0 = time()
    opt_time = 0

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

        # export mesh
        if cpp_compiled:
            mt.export_vmesh(V.points.cpu().detach().numpy(),
                            V.values.cpu().detach().numpy(), 'temp')
            os.system(
                'src/cpp_utils/build/voromesh temp.vmesh {}/{}.obj'.format(out_dir, model_name[:-4]))
        else:
            mt.export_obj(
                *V.to_mesh(), '{}/{}'.format(out_dir, model_name[:-4]))

    if cpp_compiled:
        os.system('rm -rf temp.vmesh')
    print('TOTAL TIME: {}, OPT_TIME: {}'.format(time()-T0, opt_time))
