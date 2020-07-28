'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code of HairNet.
Last modified by rqm @22:58, 2nd Aug, 2019
'''
import argparse
from model import train, test, demo

parser = argparse.ArgumentParser(description='This is the implementation of HairNet by Qiao-Mu(Albert) Ren using Pytorch.')

parser.add_argument('--mode', type = str, default='train')
parser.add_argument('--path', type = str, default='.')
parser.add_argument('--weight', type = str, default='./weight/000001_weight.pt')
parser.add_argument('--epoch', type=str, default=None)
args = parser.parse_args()

def main():
    if args.mode == 'train':
        print(args.path)
        train(args.path, args.epoch)
    if args.mode == 'test':
        test(args.path, args.weight)
    if args.mode == 'demo':
        demo(args.path, args.weight)

if __name__ == '__main__':
    main()


