import torch
import argparse
from models.config import config as config_args
from models.deeppruner import DeepPruner
from dataloader import sceneflow_collector as lt
from dataloader import sceneflow_loader as DA

def evaluate_time(Net,train_loader,device,**kwargs):
    import time

    Net = Net.to(device)

    for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
        imgL, imgR = imgL.to(device), imgR.to(device)
        break

    for i in range(10):
        preds = Net(imgL, imgR)

    times = 30
    start = time.perf_counter()
    for i in range(times):
        preds = Net(imgL, imgR)
    end = time.perf_counter()

    avg_run_time = (end - start) / times

    return avg_run_time
    

def step_time(args,Net,train_loader,val_loader,**kwargs):
    assert args.batch_size == 1

    Net = Net.to(args.device)

    for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
        imgL, imgR = imgL.to(args.device), imgR.to(args.device)
        break

    for i in range(10):
        preds = Net(imgL, imgR)

    Net.t.reset()

    for i in range(30):
        preds = Net(imgL, imgR)

    print(Net.t.all_avg_time_str(30))

def flops(Net,device):
    Net = Net.to(device)
    input = torch.randn(1,3,256,512).to(device)

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net, (input, input))   # FLOPs（乘加=2）
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
    parser.add_argument('--device', default='cpu', type=str)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.cost_aggregator_scale = config_args.cost_aggregator_scale
    args.maxdisp = config_args.max_disp

    #Dataset
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath,)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.SceneflowLoader(all_left_img, all_right_img, all_left_disp, args.cost_aggregator_scale*8.0, True),
        batch_size=1, shuffle=True, num_workers=8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.SceneflowLoader(test_left_img, test_right_img, test_left_disp, args.cost_aggregator_scale*8.0, False),
        batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    #model
    Net = DeepPruner()

    avg_run_time = evaluate_time(args=args,Net=Net,train_loader=TrainImgLoader,device=args.device)
    total_flops,total_params = flops(Net,args.device)

    print(avg_run_time)
    print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")