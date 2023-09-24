import re
import time
import json
import pickle
import random
import argparse
import numpy as np
from utils import *
from generation import *
from tqdm import tqdm
from data_utils import StrategyQA, GSM8k, Aqua, ECQA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--round', default=2, type=int)
    args = parser.parse_args()

    if args.dataset == "SQA":
        data = StrategyQA(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "ECQA":
        data = ECQA(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "GSM8k":
        data = GSM8k(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "Aqua":
        data = Aqua(data_dir=f'./dataset/{args.dataset}')

    test_samples = data.get_test_samples()[:args.num_samples]
    print(f"Number of test samples={len(test_samples)}")

    with open(f'convincing/{args.dataset}/chatgpt.json', 'r') as f:
        convincing_gpt = json.load(f)
    with open(f'convincing/{args.dataset}/claude.json', 'r') as f:
        convincing_claude = json.load(f)
    with open(f'convincing/{args.dataset}/bard.json', 'r') as f:
        convincing_bard = json.load(f)

    claude = ClaudeModel()

    # Phase1: Initial Response Generation

    claude_result = []
    while True:
        for test_sample in tqdm(test_samples[len(claude_result):]):
            tmp = {}
            tmp['gold_answer'] = test_sample['answer']
            try:
                result = claude.claude_gen_ans(test_sample,
                                            convincing_samples=convincing_gpt+convincing_bard,
                                            additional_instruc=None,
                                            intervene=False,
                                            dataset=args.dataset)
            except ValueError:
                print("cannot generate valid response for this sample.")
                result = invalid_result(args.dataset)
            if result == 403:
                pause = input("rate limit: let's take a break. enter anything to resume: ")
                if pause: break

            tmp['prediction'] = result
            claude_result.append(tmp)
            time.sleep(1)
        break

    gpt_result = []
    for test_sample in tqdm(test_samples[len(gpt_result):]):
        tmp = {}
        tmp['gold_answer'] = test_sample['answer']
        try:
            result = gpt_gen_ans(test_sample,
                                convincing_samples=convincing_claude+convincing_bard,
                                additional_instruc=None,
                                intervene=False,
                                dataset=args.dataset)
        except InvalidRequestError:
            print("blocked by Azure OpenAI’s content management policy.")
            result = invalid_result(args.dataset)
        tmp['prediction'] = result
        gpt_result.append(tmp)
        time.sleep(1)
   
    bard_result = []
    for test_sample in tqdm(test_samples[len(bard_result):]):
        tmp = {}
        tmp['gold_answer'] = test_sample['answer']
        try:
            result = bard_gen_ans(test_sample,
                                convincing_samples=convincing_claude+convincing_gpt,
                                additional_instruc=None,
                                intervene=False,
                                dataset=args.dataset)   
            tmp['prediction'] = result
        except ValueError:
            tmp['prediction'] = invalid_result(args.dataset)
        
        bard_result.append(tmp)
        time.sleep(1)

    # Evaluation for the initial round

    all_results = []
    for c, g, b in zip(claude_result, gpt_result, bard_result):
        tmp = {}
        tmp['gold_answer'] = c['gold_answer']
        tmp['claude_output_0'] = c['prediction']
        tmp['gpt3_output_0'] = g['prediction']
        tmp['bard_output_0'] = b['prediction']
        all_results.append(tmp)

    all_results = clean_output(all_results, 0, dataset=args.dataset)
    all_results = parse_output(all_results, 0)
    print(f"Initial Round Performance: {evaluate_all(all_results, 0)}")

    # Phase2: Multi-Round Discussion

    for r in range(1, args.round+1):
        print(f"----- Round {r} Discussion -----")
        all_results = claude.claude_debate(test_samples,
                                        all_results,
                                        rounds=r,
                                        convincing_samples=convincing_gpt+convincing_bard,
                                        dataset=args.dataset)

        all_results = gpt_debate(test_samples,
                            all_results,
                            rounds=r,
                            convincing_samples=convincing_claude+convincing_bard,
                            dataset=args.dataset)

        all_results = bard_debate(test_samples,
                            all_results,
                            rounds=r,
                            convincing_samples=convincing_claude+convincing_gpt,
                            dataset=args.dataset)

        all_results = clean_output(all_results, r, dataset=args.dataset)
        all_results = parse_output(all_results, r)
        print(f"Round {r} Performance: {evaluate_all(all_results, r)}")

    with open(f'{args.dataset}_round_{args.round}.pkl', 'wb') as f:
        pickle.dump(all_results, f)