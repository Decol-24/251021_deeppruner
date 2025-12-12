import torch
import argparse
from models.config import config as config_args
from models.deeppruner import DeepPruner
from dataloader import sceneflow_collector as lt
from dataloader import sceneflow_loader as DA

@torch.no_grad()
def evaluate_time(Net,imgL,imgR,device,**kwargs):
    import time

    Net = Net.to(device)
    imgL = imgL.to(device)
    imgR = imgR.to(device)

    for i in range(10):
        preds = Net(imgL, imgR)

    times = 30
    start = time.perf_counter()
    for i in range(times):
        preds = Net(imgL, imgR)
    end = time.perf_counter()

    avg_run_time = (end - start) / times

    return avg_run_time

@torch.no_grad()
def evaluate_flops(Net,input,device,**kwargs):
    Net = Net.to(device)
    # input = input.to(device)

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net,input)   # FLOPs（乘加=2）
    total_flops = flops.total()

    total_params = sum(p.numel() for p in Net.parameters())
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")

    return total_flops,total_params


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepPruner')
    parser.add_argument('--datapath', default='/home/liqi/Code/Scene_Flow_Datasets/',
                        help='datapath for sceneflow dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    parser.add_argument('--save_dir', default='./result/',
                        help='save directory')
    parser.add_argument('--savemodel', default='./',
                        help='save model')
    parser.add_argument('--logging_filename', default='./train_sceneflow.log',
                        help='save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    args.cost_aggregator_scale = config_args.cost_aggregator_scale
    args.maxdisp = config_args.max_disp

    #model
    Net = DeepPruner()

    Net = Net.to(args.device)
    imgL = torch.randn(1,3,544,960).to(args.device)
    imgR = torch.randn(1,3,544,960).to(args.device)

    avg_run_time = evaluate_time(Net=Net,imgL=imgL,imgR=imgR,device=args.device)
    total_flops,total_params = evaluate_flops(Net,input=(imgL,imgL),device=args.device)

    print(avg_run_time)
    print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")