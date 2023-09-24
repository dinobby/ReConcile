import os
import re
import ast
import json
import time

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

random.seed(1234)
datasets = ["SQA", "GSM8k", "ECQA", "Aqua"]

def prepare_context(test_sample, convincing_samples=None, intervene=False, dataset="SQA"):
    assert dataset in datasets
    context = []
    if convincing_samples:
        for cs in convincing_samples:
            context.append(f"Q: {cs['train_sample']['question']}\nA:" + str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']}))
     
    if intervene:
        context.append("Q: " + test_sample['question'] + "\nAnswer the question given the fact that " + test_sample['gold_explanation'])
    else:
        context.append("Q: " + test_sample['question'])

    if dataset in ["ECQA", "Aqua", "ScienceQA"]:
        context.append("The options are: {}. Please select an option as your answer.".format(" ".join(test_sample["options"])))
        
    context.append("Please answer the question with step-by-step reasoning.")
    context.append("\nAlso, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right.")
    context.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
    if dataset == "SQA":
        context.append("Only answer yes or no in the \"answer\" field.")
    elif dataset == "GSM8k":
        context.append("Only place a single numeric value in the \"answer\" field.")
    elif dataset == "ECQA":
        context.append("Only place 1,2,3,4,5 representing your choice in the \"answer\" field.")
    elif dataset == "Aqua":
        context.append("Only place A,B,C,D,E representing your choice in the \"answer\" field.")  
    context.append("Do not output irrelevant content.")
    return "\n".join(context)
    
def prepare_context_for_chat_assistant(sample, convincing_samples=None, intervene=False, dataset="SQA"):
    assert dataset in datasets
    contexts = []
    if convincing_samples:
        for cs in convincing_samples:
            contexts.append({"role": "user", "content": f"Q: {cs['train_sample']['question']}"})
            contexts.append({"role": "assistant", "content": str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']})})

    if intervene:
        contexts.append({"role": "user", "content": f"Q: {sample['question']}" + "\nAnswer the question given the fact that " + sample['gold_explanation']})  
    else:
        contexts.append({"role": "user", "content": f"Q: {sample['question']}"})
        
    if dataset in ["ECQA", "Aqua"]:
        contexts[-1]["content"] += "The options are: {}. Please select an option as your answer.".format(" ".join(sample["options"]))
        
    contexts[-1]["content"] += " Please answer the question with step-by-step reasoning."
    contexts[-1]["content"] += " Also, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right."
    contexts[-1]["content"] += " Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format."
    
    if dataset == "SQA":
        contexts[-1]["content"] += " Only answer yes or no in the \"answer\" field."
    elif dataset == "GSM8k":
        contexts[-1]["content"] += " Only place a single numeric value in the \"answer\" field."
    elif dataset == "ECQA":
        contexts[-1]["content"] += " Only place 1,2,3,4,5 representing your choice in the \"answer\" field."
    elif dataset == "Aqua":
        contexts[-1]["content"] += " Only place A,B,C,D,E representing your choice in the \"answer\" field."    
    contexts[-1]["content"] += " Do not output irrelevant content."
    return contexts

def prepare_context_for_bard(test_sample, convincing_samples=None, intervene=False, dataset="SQA"):
    assert dataset in datasets
    context, convincing_icx, unhelpful_icx = [], [], []
    if convincing_samples:
        for cs in convincing_samples:
            convincing_icx.append((f"Q: {cs['train_sample']['question']}", str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']})))
    
    if intervene:
        context.append("Q: " + test_sample['question'] + "\nAnswer the question given the fact that " + test_sample['gold_explanation'])
    else:
        context.append("Q: " + test_sample['question'])
        
    if dataset in ["ECQA", "Aqua"]:
        context.append("The options are: {}. Please select an option as your answer.".format(" ".join(test_sample["options"])))  
        
    context.append("Please answer the question with step-by-step reasoning.")
    context.append("Also, evaluate your confidence level (between 0.0 and 1.0) to indicate the possibility of your answer being right.")
    context.append("Output your answer in json format, with the format as follows: {\"reasoning\": \"\", \"answer\": \"\", \"confidence_level\": \"\"}. Please strictly output in JSON format.")
    
    if dataset == "SQA":
        context.append("Only answer yes or no in the \"answer\" field. Do not output irrelevant content.")
    elif dataset == "GSM8k":
        context.append("Only place a single numeric value in the \"answer\" field. Do not output irrelevant content.")
    elif dataset == "ECQA":
        context.append("Only place 1,2,3,4,5 representing your choice in the \"answer\" field.")
    elif dataset == "Aqua":
        context.append("Only place A,B,C,D,E representing your choice in the \"answer\" field.")
    context.append("Do not output irrelevant content.")
    return "\n".join(context), convincing_icx, unhelpful_icx

def invalid_result(dataset):
    if dataset == "SQA":
        result = {"reasoning": "",
          "answer": np.random.choice(['yes', 'no']),
          "confidence_level": 0.0}
    elif dataset=="GSM8k":
        result = {"reasoning": "",
          "answer": "0",
          "confidence_level": 0.0}
    elif dataset=="ECQA":
        result = {"reasoning": "",
          "answer": np.random.choice(['1', '2', '3', '4', '5']),
          "confidence_level": 0.0}   
    elif dataset=="Aqua":
        result = {"reasoning": "",
          "answer": np.random.choice(['A', 'B', 'C', 'D', 'E']),
          "confidence_level": 0.0}
    return result

def parse_json(model_output):
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    try:
        model_output = model_output.replace("\n", " ")
        model_output = re.search('({.+})', model_output).group(0)
        model_output = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", model_output)
        result = ast.literal_eval(model_output)
    except (SyntaxError, NameError, AttributeError):
        return "ERR_SYNTAX"
    return result

def find_idx_by_element(input_list, element):
    return [i for i, a in enumerate(input_list) if a == element]

def find_element_by_indices(input_list, index_list):
    return [b for i, b in enumerate(input_list) for k in index_list if i == k]

def trans_confidence(x):
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1

def parse_output(all_results, rounds):
    c, g, b = "claude", "gpt3", "bard"
    r = "_output_"+str(rounds)
    
    for i in all_results:
        certainty_vote = {}
            
        for o in [c, g, b]:
            if o+r in i:
                i[o+"_pred_"+str(rounds)] = i[o+r]['answer']
                i[o+"_exp_"+str(rounds)] = f"I think the answer is {i[o+r]['answer']} because {i[o+r]['reasoning']} My confidence level is {i[o+r]['confidence_level']}." 
                if i[o+r]['answer'] not in certainty_vote:
                    certainty_vote[i[o+r]['answer']] = trans_confidence(i[o+r]['confidence_level']) + 1e-5
                else:
                    certainty_vote[i[o+r]['answer']] += trans_confidence(i[o+r]['confidence_level'])
        if c+r in i and g+r in i and b+r in i:
            i['vote_'+str(rounds)] = [i['claude_pred_'+str(rounds)], i['gpt3_pred_'+str(rounds)], i['bard_pred_'+str(rounds)]]
            i['exps_'+str(rounds)] = [i['claude_exp_'+str(rounds)], i['gpt3_exp_'+str(rounds)], i['bard_exp_'+str(rounds)]]
            i['weighted_vote_'+str(rounds)] = certainty_vote
            i['weighted_max_'+str(rounds)] = max(certainty_vote, key=certainty_vote.get)

            i['debate_prompt_'+str(rounds)] = ''
            vote = Counter(i['vote_'+str(rounds)]).most_common(2)
            i['majority_ans_'+str(rounds)] = vote[0][0]
            if len(vote) > 1: # not all the agents give the same answer
                for v in vote:
                    i['debate_prompt_'+str(rounds)] += f"There are {v[1]} agents think the answer is {v[0]}. "
                    exp_index = find_idx_by_element(i['vote_'+str(rounds)], v[0])
                    group_exp = find_element_by_indices(i['exps_'+str(rounds)], exp_index)
                    exp = "\n".join(["One agent solution: " + g for g in group_exp])
                    i['debate_prompt_'+str(rounds)] += exp + "\n\n"
                    
    return all_results

def evaluate_single_model(results):
    num_correct = 0
    for i in results:
        if i['gold_answer'] == i['prediction']['answer']:
            num_correct+=1
    return num_correct / len(results)

def clean_output(all_results, rounds, dataset):
    co, go, bo = "claude_output_" + str(rounds), 'gpt3_output_' + str(rounds), 'bard_output_' + str(rounds)
    for i in all_results:
        for o in [co, go, bo]:
            if o in i:
                    
                if 'reasoning' not in i[o]:
                    i[o]['reasoning'] = ""
                elif type(i[o]['reasoning']) is list:
                    i[o]['reasoning'] = " ".join(i[o]['reasoning'])

                if dataset=="SQA":
                    if 'answer' not in i[o] or i[o]['answer'] not in ['yes', 'no']:
                        i[o]['answer'] = np.random.choice(['yes', 'no'])
                elif dataset=="GSM8k":
                    if 'answer' not in i[o]:
                        i[o]['answer'] = "0"
                elif dataset=="ECQA":
                    if 'answer' not in i[o]:
                        i[o]['answer'] = np.random.choice(['1', '2', '3', '4', '5'])
                elif dataset=="Aqua":
                    if 'answer' not in i[o]:
                        i[o]['answer'] = np.random.choice(['A', 'B', 'C', 'D', 'E'])
                        
                if 'confidence_level' not in i[o] or not i[o]['confidence_level']:
                    i[o]['confidence_level'] = 0.0
                else:
                    if type(i[o]['confidence_level']) is str and "%" in i[o]['confidence_level']:
                            i[o]['confidence_level'] = float(i[o]['confidence_level'].replace("%","")) / 100
                    else:
                        try:
                            i[o]['confidence_level'] = float(i[o]['confidence_level'])
                        except:
                            print(i[o]['confidence_level'])
                            i[o]['confidence_level'] = 0.0
                
    return all_results

def evaluate_results(all_results, prefix, rounds):
    num_correct = 0
    for i in all_results:
        r_num = int(rounds)
        while True:
            eval_key = prefix+"_"+str(r_num)
            if eval_key in i:
                if i['gold_answer'] == i[eval_key]:
                    num_correct += 1
                break
            else:
                r_num -= 1
            if r_num<0: break
    return num_correct/len(all_results)

def evaluate_all(all_results, rounds):
    accuracies = {}
    for prefix in ["claude_pred", "gpt3_pred", "bard_pred", "majority_ans", "weighted_max"]:
        accuracies[prefix] = evaluate_results(all_results, prefix, rounds)
    return accuracies

