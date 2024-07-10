# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch

from mmengine.analysis import get_model_complexity_info

from mmpretrain import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--warmup', 
        type=int,
        default=10,
        help='number of warmup iterations for fps calculation'
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=10,
        help='number of iterations for fps calculation'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = get_model(args.config)
    model.eval()
    # if hasattr(model, 'extract_feat'):
    #     model.forward = model.extract_feat
    # else:
    #     raise NotImplementedError(
    #         'FLOPs counter is currently not currently supported with {}'.
    #         format(model.__class__.__name__))
    analysis_results = get_model_complexity_info(
        model,
        input_shape,
    )
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    activations = analysis_results['activations_str']
    out_table = analysis_results['out_table']
    out_arch = analysis_results['out_arch']
    # print(out_arch)
    # print(out_table)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n'
          f'Activation: {activations}')
    # print('!!!Only the backbone network is counted in FLOPs analysis.')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')

    random_inputs = torch.randn(1, *input_shape, requires_grad=False)
    for _ in range(args.warmup):
        model(random_inputs)

    time_list_cpu = []
    
    with torch.no_grad():
        for _ in range(args.iter):
            tic = time.perf_counter()
            model(random_inputs)
            time_list_cpu.append(time.perf_counter() - tic)

    time_list_cpu = torch.tensor(time_list_cpu)
    fps = 1 / time_list_cpu
    print(f'FPS: {torch.mean(fps):.2f}±{torch.std(fps):.2f} (CPU)')

    if torch.cuda.is_available():
        model.cuda()
        random_inputs = random_inputs.cuda()
        time_list_gpu = []

        for _ in range(args.warmup):
            model(random_inputs)

        with torch.no_grad():
            for _ in range(args.iter):
                torch.cuda.synchronize()
                tic = time.perf_counter()
                model(random_inputs)
                torch.cuda.synchronize()
                time_list_gpu.append(time.perf_counter() - tic)


        time_list_gpu = torch.tensor(time_list_gpu)
        fps = 1 / time_list_gpu
        print(f'FPS: {torch.mean(fps):.2f}±{torch.std(fps):.2f} (GPU)')

    print(split_line)

if __name__ == '__main__':
    main()
