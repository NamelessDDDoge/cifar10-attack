import argparse
import os
import torch.utils
import tqdm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import transferattack
from transferattack.utils import *


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='awt', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=40, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 /255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 /255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=1, type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='1', type=str)
    parser.add_argument('--awt', action='store_true', help='enable awt')
    return parser.parse_args()

def generate_adversarial_samples(args: argparse.ArgumentParser, dataloader: torch.utils.data.DataLoader):
    # Generate adversarial samples
    if args.ensemble or len(args.model.split(',')) > 1:
        args.model = args.model.split(',')
    attacker = transferattack.load_attack_class(args.attack)(
        model_name=args.model, 
        targeted=args.targeted,  
        epsilon=args.eps, 
        alpha=args.alpha, 
        epoch=args.epoch, 
        decay=args.momentum,
        awt=args.awt,
        lr = 0.001, rho=0.002,
        random_start=True
        )

    for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
        perturbations = attacker(images, labels)
        save_images(args.output_dir, images + perturbations.cpu(), filenames)

def evaluate_adversarial_samples(args: argparse.ArgumentParser, iteration: int, dataloader: torch.utils.data.DataLoader):
    asr = {}
    for model_name, model in load_pretrained_model(cnn_model, vit_model):
        model = wrap_model(model.eval().cuda())
        for p in model.parameters():
            p.requires_grad = False
        correct, total = 0, 0
        for images, labels, _ in dataloader:
            if args.targeted:
                labels = labels[1]
            pred = model(images.cuda())
            correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
            total += labels.shape[0]
        if args.targeted:
            asr[model_name] = (correct / total) * 100
        else:
            asr[model_name] = (1 - correct / total) * 100
        print(f"Iteration {iteration}: {model_name} ASR: {asr[model_name]:.1f}%")
    
    return asr


def main():
    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval, sample_num=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
    # generate adversarial samples

    model_list = [item for sublist in [cnn_model, vit_model] for item in sublist]
    results = {model_name: [] for model_name in model_list}

    for i in range(1):
        dataset.train()
        generate_adversarial_samples(args, dataloader)
        dataset.eval()
        asr = evaluate_adversarial_samples(args, i, dataloader)
        for model_name in asr.keys():
            results[model_name].append(asr[model_name])

    data_records = []
    analyze_records = []

    analyze_dict = {
        'model': args.model,
        'attack': args.attack + "_awt" if args.awt else args.attack,
    }
    for model_name in asr.keys():
        mean_asr = np.mean(results[model_name])
        var_asr = np.var(results[model_name])
        
        data_records.append({
            'model': args.model,
            'attack': args.attack + "_awt" if args.awt else args.attack,
            'check_model': model_name, 
            'mean ASR': mean_asr, 
            'variance ASR': var_asr
            })
        
        analyze_dict[model_name] = f'{mean_asr:.1f}±{var_asr:.1f}'

        print(f"Final Results for {model_name}: Mean ASR: {mean_asr:.1f}%, Variance: {var_asr:.1f}%")

    analyze_records.append(analyze_dict)

    # Save results to CSV
    data_df = pd.DataFrame(data_records)
    data_df.to_csv('data_record.csv', index=False, mode='a', header=not os.path.isfile('data_record.csv'))
    print("Results saved to data_record.csv")

    analyze_df = pd.DataFrame(analyze_records)
    analyze_df.to_csv('analyze_table.csv', index=False, mode='a', header=not os.path.isfile('analyze_table.csv'))
    print("Results saved to analyze_table.csv")
    

if __name__ == '__main__':
    main()
