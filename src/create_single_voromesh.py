import utils.mesh_tools as mt
import utils.custom_voronoi as cv
import torch
import argparse
from time import time


DEFAULTS = {
    "grid_n": 32,
    "samples_fac": 150,
    "lr": 5e-3,
    "epochs": 400,
    "random_mask": .2
}


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Creates a VoroMesh from a given mesh. The user can choose any grid size, the other hyperparameters will adjust to it')
    parser.add_argument('shape_path', type=str, help='path to input mesh')
    parser.add_argument('--output_name', type=str,
                        default='voromesh', help='name of output mesh')
    parser.add_argument('--export_vmesh', type=bool,
                        default=False, help='export generators in .vmesh for precise mesh extraction')
    parser.add_argument('--grid_n', type=int,
                        default=DEFAULTS["grid_n"], help='grid_size')
    parser.add_argument('--samples_fac', type=int, default=DEFAULTS["samples_fac"],
                        help='total number of samples: grid_n**2*samples_fac')
    parser.add_argument('--lr', type=float,
                        default=DEFAULTS["lr"], help='learning rate')
    parser.add_argument('--epochs', type=int,
                        default=DEFAULTS["epochs"], help='epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--random_mask', type=float, default=DEFAULTS["random_mask"],
                        help='proportion of subsampled points for each epoch')
    parser.add_argument('--lr_scheduler', type=bool,
                        default=True, help='decaying lr')
    return parser


def create_voromesh(samples, v, f, grid_n=DEFAULTS["grid_n"], lr=DEFAULTS["lr"], epochs=DEFAULTS["epochs"], device='cuda', random_mask=DEFAULTS["random_mask"], lr_scheduler=True, return_time=False):
    mgrid = mt.mesh_grid(grid_n, True)
    V = cv.VoronoiValues(mgrid)
    # subselect points close to the samples
    V.subselect_cells(cv.mask_relevant_voxels(grid_n, samples))
    tensor_points = torch.tensor(samples, dtype=torch.float32).to(device)
    V.to(device)
    optimizer = torch.optim.Adam(V.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 150, 200, 250], gamma=0.5)
    t0 = time()
    for _ in (range(epochs)):
        cv.train_voronoi(V, tensor_points, optimizer, random_mask)
        if lr_scheduler:
            scheduler.step()
    t0 = time()-t0
    V.set_values_winding(v, f, True)
    if return_time:
        return V, t0
    return V


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    v, f, samples = mt.load_and_sample_shape(
        args.shape_path, '', args.samples_fac*args.grid_n**2)
    V = create_voromesh(samples, v, f, args.grid_n, args.lr,
                        args.epochs, args.device, args.random_mask, args.lr_scheduler)
    mt.export_obj(*V.to_mesh(), args.output_name)
    if args.export_vmesh:
        mt.export_vmesh(V.points.cpu().detach().numpy(),
                        V.values.cpu().detach().numpy(), args.output_name)
