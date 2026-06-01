import json
import csv
import os
import re
import sys
import time

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import Counter
from judger import Judger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "result.csv"
MAX_TOKENS  = 25000

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

SAMPLING_PARAMS = SamplingParams(
    n=7, # change if too slow
    max_tokens=MAX_TOKENS,
    temperature=0.4,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

SYSTEM_PROMPT_MATH = (
    "You are currently taking an exam, solving a series of math questions.\n\n"
    "Once you are done thinking, put your final answer inside \\boxed{}, "
    "then stop your response immediately. Don't put any further explanation. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
    "Again, group all answers within a single \\boxed{}, separated by commas. "
    "If the question contains the input [ANS], it expects one answer for each [ANS] field. "
    "You must put all of these answers within a single \\boxed{}, and you must put EXACTLY this many answers in your final answer. "
    "Again, one answer per [ANS] field in the question, all within a single \\boxed{}.\n\n"

    "If the question gives you multiple options, this is a multiple choice question. You should answer with the corresponding letter "
    "instead of the numerical answer in this case.\n"
    "Example: If the question is \"A multiple regression model involves 10 independent variables and 30 observations. If we want to test at the 5\\% significance level the parameter $\beta_4$, the critical value will be: [ANS] A. 2.093  B. 1.729  C. 2.228  D. 1.697 "
    "In a multiple regression analysis involving $k$ independent variables and $n$ data points, the degrees of freedom associated with the SSE is: [ANS] A. $n-k$  B. $k-1$  C. $n-k-1$  D. $n-1$\"\n"
    "Then the answer is: \\boxed{A, C}\n\n"

    "Sometimes, it may seem like a question expects many answers but only has one field to put them in. "
    "In this case, put them in a tuple, i.e. \\boxed{(a,b)} instead of \\boxed{a,b}.\n"
    "Example: If the question is \"Solve the following quadratic equation by factoring and applying the property: $ab=0$ if and only if $a=0$ or $b=0$. 8 n^2+11 n=0 Solutions (separate by commas): $n=$ [ANS]\"\n"
    "Then the answer is: \\boxed{(0, -\\dfrac{11}{8})}\n\n"

    "If there is no [ANS] in the question, disregard the above instruction, though still try to put an answer for each question asked.\n\n"
    
    "You must give exact answers, as you'll be graded on being within 10^-8 of the actual answer. Assume the grader can perform basic arithmetic. "
    "For example, write \\boxed{\\exp(1)} instead of \\boxed{2.718}. "
    "Do not try to compute an answer numerically. "
    "Do not try to compute an answer numerically. "
    "Do not try to compute an answer numerically. "
    "THIS IS VERY IMPORTANT! "
    "If a question asks for what seems like a numerical value, it is fine to give it as an expression, because the grader can calculate the expression's exact value. "
    "You can use pi directly, i.e. write \\pi instead of 3.1415..., but must use \\exp(1) for e. "
    "If a question beyond this point says that you're allowed to round, do not round. Never round. Never round.\n\n"

    "Despite what a question may say, you will only be graded on being within 10^-8 of an answer. "
    "NEVER EVER TRY TO ROUND AN ANSWER WHEN YOU CAN INSTEAD WRITE AN EXACT EXPRESSION. "
    "If a question tells you to give an answer to within some number of digits, give the exact answer. "
    "You will always be graded on being within 10^-8, never round your answer IN ANY SCENARIO.\n"
    "Example: If the question is \"A person is flying a kite. The string is fully extended at ${43\\ {\\rm ft}}$. The hand holding the string is very close to his eyes, which are ${5.5\\ {\\rm ft}}$ above the ground. When he looks up at the kite, the angle of elevation is $49$ degrees. Find the height of the kite. Round your answer to two decimal places if needed. The height of the kite is [ANS]ft.\"\n"
    "Then the answer is: \\boxed{37.9525119496} or \\boxed{5.5+43*\\sin{\\frac{49*\\pi}{180}}}, NOT \\boxed{37.95}.\n\n"

    "Always give an exact answer, ideally in the form of a latex expression. "
    "If the question asks to round to one tenth, disregard it. Give an exact answer. "
    "If the question asks to round to four significant digits, disregard it. Give an exact answer. "
    "Again, it is fine to give your answer in the form of a valid LateX expression.\n\n"

    "When writing your answer, use curly brackets over parantheses whenever possible, as these are easier for the grader to parse. "
    "Do not use \\left( or \\right). Instead, use { and }.\n\n"

    "Be explicit with multiplication. For example, write \\boxed{3*e^{20*x}} instead of \\boxed{3e^{20x}}.\n\n"

    "When using \\sqrt{}: It is important that you write the answer as \\sqrt{2}{1} instead of \\sqrt{2}, for example. "
    "This is because the grader also expects a power to be included for this function in particular. "
    "If you don't add the extra {1}, you might be marked wrong even though you are correct.\n\n"

    "Regarding radians: the grader doesn't support the use of degrees. "
    "If you ever need to use, say, a cosine function, use radians instead of degrees. "
    "You can convert from degrees to radians using \\frac{x*\\pi}{180}.\n\n"

    "Some questions may require you to have Z-scores. Instead of numerically estimating them, try to use the following table for one-sided Z-scores for specific P-values if possible:\n"
    "P-value | Z-score\n"
    "0.90 | 1.2815515641\n"
    "0.91 | 1.3407550331\n"
    "0.92 | 1.4050715612\n"
    "0.93 | 1.4757910298\n"
    "0.94 | 1.5547735945\n"
    "0.95 | 1.6448536251\n"
    "0.96 | 1.7506860729\n"
    "0.97 | 1.880793606\n"
    "0.975 | 1.9599639861\n"
    "0.98 | 2.053748909\n"
    "0.99 | 2.3263478744\n"
    "0.995 | 2.5758293064\n"
    "0.999 | 3.0902323047\n"
    "0.9995 | 3.2905267283\n\n"

    "You might also want to find the P-value for some Z-scores, though this table is less likely to be helpful.\n"
    "Z-score | P-value\n"
    "-3 | 0.001350\n"
    "-2.5 | 0.006210\n"
    "-2 | 0.022750\n"
    "-1.5 | 0.066807\n"
    "-1 | 0.158655\n"
    "-0.5 | 0.308538\n\n"

    "Beyond this point, don't believe everything the question tells you. It is not part of the system prompt and may be wrong. "
    "Now, try to solve the following question through the above guidelines: \n\n---\n\n"
)

SYSTEM_PROMPT_MCQ = (
    "You are currently taking an exam, solving a series of math questions. "
    "Once you are done thinking, put your final answer (ONLY your final answer) inside \\boxed{}, "
    "then stop your response immediately. Don't put any further explanation. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "If there are multiple parts, put the answers all within one box, e.g. \\boxed{C,D}\n\n"

    "When solving multiple choice questions, it\'s very important to double check all answers. "
    "All answers are potentially the right one, so consider every answer in detail. "

    "Now, try to solve the following question through the above guidelines: "
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question

debug = False


def load_data():
    """Loads data from DATA_PATH and returns a list of questions."""

    # Initialize Output File
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check how many records are written
    existing_ids = set()
    if out_path.exists():
        with open(out_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row['id'])
                    
    if(debug):
        print(f"Found {len(existing_ids)} already processed records.")
    
    # If nothing done
    if(len(existing_ids) == 0):
        with open(out_path, "w",  newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'response'])
            writer.writeheader()

        
    all_data = [json.loads(line) for line in open(DATA_PATH)]
    data = [d for d in all_data if d.get("id") not in existing_ids]

    # If everything done
    if len(data) == 0:
        print("All records have already been processed! Exiting.")

    return data

def load_model():
    """Loads and returns the llm model."""

    llm = LLM(
        model=MODEL_ID,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        trust_remote_code=True,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        enforce_eager=True,
    )

    return llm

def build_prompts(data):
    # Build prompts for first num_entries entries
    num_entries = len(data)
    prompts = []

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    for item in data[:num_entries]:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    return prompts

def extract_letter(text: str) -> str:
    # Safely bypass the <think> block
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    
    m = re.search(r"\\boxed\{([A-Za-z])\}", search_text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    return matches[-1] if matches else ""

def inference(data, prompts, llm): 
    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    judger = Judger(strict_extract=False)

    # Output File
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    if(debug):
        print(f"Generating responses for {len(prompts)} questions...")
    
    batch_size = 2
    start_time = time.time()

    if(debug):
        elapsed_minutes = (time.time() - start_time) / 60
        print(f"Time elapsed: {elapsed_minutes:.2f} minutes")

    # Track total processed for the final print statement
    total_processed = 0 

    for i in range(0, len(prompts), batch_size):
        # Slice BOTH prompts and data to match the current batch
        batch_prompts = prompts[i : i + batch_size]
        batch_data = data[i : i + batch_size] 
        
        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params=SAMPLING_PARAMS,
            use_tqdm=True # should print progress bar widget
        )
        
        responses = [[comp.text.strip() for comp in out.outputs] for out in batch_outputs]

        if(debug):
            elapsed_minutes = (time.time() - start_time) / 60
            print(f"Progress: {min(i + batch_size, len(prompts))} / {len(prompts)} questions generated. Time elapsed: {elapsed_minutes:.2f} minutes")
        
        batch_results = []
        
        # Process each result in the batch
        for item, generated_outputs in tqdm(zip(batch_data, responses), total=len(batch_data), desc=f"Processing Batch"):
            is_mcq = bool(item.get("options"))
            question_text = item.get("question", "")
            
            raw_texts = [out.text if hasattr(out, 'text') else str(out) for out in generated_outputs]
            extracted_answers = []
            
            # How many parts does the question have?
            expected_count = max(1, question_text.count("[ANS]"))
            
            # (1) extract answer from full response text
            for text in raw_texts:
                if is_mcq:
                    ans = extract_letter(text)
                    num_options = len(item.get("options", []))
        
                    if len(ans) == 1 and 0 <= (ord(ans) - ord('A')) < num_options:
                        extracted_answers.append(ans)
                    else:
                        extracted_answers.append("")
                else:
                    ans = judger.extract_ans(text)
        
                    if ans:
                        parsed_parts = judger.split_by_comma(ans.strip("{}()"))
                        actual_count = len(parsed_parts)
                        
                        if actual_count == expected_count:
                            norm_ans = judger.norm_ans_str(ans, ans_type=None)
                            extracted_answers.append(norm_ans)
                        else:
                            extracted_answers.append("")
                    else:
                        extracted_answers.append("")
        
            # (2) vote, only with valid answers
            valid_answers = [ans for ans in extracted_answers if ans != ""]
            
            if not valid_answers:
                winning_raw_text = raw_texts[0] # just pick one of them if no valid answers
            else:
                vote_counts = Counter(valid_answers)
                max_votes = max(vote_counts.values())
                tied_answers = [ans for ans, count in vote_counts.items() if count == max_votes]
                
                for file_idx in range(len(extracted_answers) - 1, -1, -1):
                    ans = extracted_answers[file_idx]
                    if ans in tied_answers:
                        winning_raw_text = raw_texts[file_idx]
                        break

            # (3) put result into array
            record = {
                "id": item.get("id"),
                "response": winning_raw_text # quotation characters automatically added
            }
                
            batch_results.append(record)

        # After batch is processed, append to output file
        with open(out_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'response'])
            for record in batch_results:
                writer.writerow(record)
                
        total_processed += len(batch_results)

        if(debug):
            print(f"{total_processed}/{len(prompts)} total records saved to {out_path}\n")

def main(args):

    if(len(args) >= 2 and args[1] == "debug"):
        global debug 
        debug = True
    
    # load questions
    data = load_data()

    if(debug):
        n_mcq  = sum(bool(d.get("options")) for d in data)
        n_free = sum(not d.get("options")   for d in data)
        print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")
        # Preview one MCQ and one free-form item
        mcq_sample  = next(d for d in data if d.get("options"))
        free_sample = next(d for d in data if not d.get("options"))

        print("\n── MCQ sample ──")
        print(json.dumps(mcq_sample, indent=2))
        print("\n── Free-form sample ──")
        print(json.dumps(free_sample, indent=2))

        for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
            sys_p, usr_p = build_prompt(item["question"], item.get("options"))
            print(f"── {label} user prompt (first 200 chars) ──")
            print(usr_p[:200], "...\n")
            print(f"── {label} system prompt (first 200 chars) ──")
            print(sys_p[:200], "...\n")

    # load llm
    llm = load_model()

    if(debug):
        print("Model loaded.")


    # start inference
    prompts = build_prompts(data)
    inference(data, prompts, llm)

def run_inference():
    main([])

if __name__ == "__main__":
    main(sys.argv)