# ReConcile: Round-Table Conference Improves Reasoning via Consensus Among Diverse LLMs
The code for ReConcile, a group-discuss-and-convince framework

[Justin Chen](https://dinobby.github.io/), [Swarnadeep Saha](https://swarnahub.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
<img width="858" alt="image" src="https://github.com/dinobby/ReConcile_bak/assets/20419883/1204ecdd-4635-4d75-8233-c00dac56b30a">

### Overview of the multi-model multi-agent discussion
<img width="809" alt="image" src="https://github.com/dinobby/ReConcile/assets/20419883/0924a818-3abb-4406-b521-7166f9d78d0f">

# Installation
This project is tested on Python 3.10.11. All dependencies can be installed via:

```pip install -r requirements.txt```

# Run
First, you will need the following keys in your ```.env``` file:

```
OPEN_AI_API_BASE = "..."
OPEN_AI_API_VERSION = "..." 
OPEN_AI_API_KEY = "..."
PALM_API_KEY = "..."
CLAUDE_COOCKIE1 = "..."
CLAUDE_COOCKIE2 = "..."
CLAUDE_COOCKIE3 = "..."
CLAUDE_COOCKIE4 = "..."
CLAUDE_COOCKIE5 = "..."
```
If you use more than five accounts for Claude, you can put more here, and remember to edit ```claude_coockies``` in ```generation.py```

Next, run the following command to start:

```
python run.py --num_samples 100 --dataset SQA
````

Currently, dataset can be ```["SQA", "GSM8k", "ECQA", "Aqua"]``` 

# Dataset
The datasets used in this work are already included in the ```dataset``` folder.
- StrategyQA: https://github.com/eladsegal/strategyqa
- ECQA: https://github.com/IBM/ecqa
- GSM8K: https://github.com/openai/grade-school-math
- AQuA: https://github.com/google-deepmind/AQuA

# Citation
```
@article{chen2023reconcile,
  title={ReConcile: Round-Table Conference Improves Reasoning via Consensus Among Diverse LLMs},
  author={Chen, Justin Chih-Yao and Saha, Swarnadeep and Bansal, Mohit},
  journal={arXiv preprint arXiv:2306.xxxxx},
  year={2023}
}
```
