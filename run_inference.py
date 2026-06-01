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
    "You are an expert math solver taking an online exam. Follow the rules strictly and carefully.\n"
    "RULES:\n"
    "1. BOXED ANSWERS: You must put all answers inside ONE single \\boxed{} tag, separated by commas (e.g., \\boxed{3, \\frac{u}{2 \\cdot k+3}}). "
    "Only the last \\boxed{} tag will be graded, so you may output more \\boxed{} tags if you want to change your answer.\n"
    "2. MULTIPLE PARTS and [ANS] FIELDS: Before working on the problem, determine exactly how many answers are expected. There is no partial credit. "
    "If the question contains [ANS] fields, you must provide exactly one answer for each [ANS] field. "
    "If there are X [ANS] fields, there must be X-1 commas inside the final \\boxed{} answer, EXCLUDING commas inside additional nested brackets. For example, \\boxed{3, (14, 15)} would count as 2 parts."
    "If an [ANS] field represents a multiple-choice question, output ONLY the capital letter(s) for that field. "
    "If multiple answers seem to fit into one [ANS] field, formatting depends on the style of question:"
    "If it is a multi-select question with letter options(e.g., 'Select every formula'), concatenate your choices into a single string without spaces or commas (e.g., \\boxed{BEF}). "
    "If it is a free-response question, (e.g. find roots of a polynomial in a single [ANS]), surround your multiple answers in these brackets (). (e.g. \\boxed{(2, 3)}). "
    "Note commas placed in these brackets () don't count towards adding a new part.\n"
    "3. EXACT EXPRESSIONS: Unlike traditional exams, your answer will be parsed with sympy, a latex parser. "
    "IMPORTANT (PRECISION): if the answer expects a number, you should always output an equivalent arithmetic expression, e.g. \\boxed{\\exp(1) + \\pi + 3^3}. "
    "ALWAYS PREFER EXACT EXPRESSIONS OVER NUMERIC VALUES, even if the question says e.g. your answer is [ANS] farenheit. "
    "For example, write \\boxed{75 + 110 \\cdot {\\frac{8}{11}}^{\\frac{3}{2}}} instead of \\boxed{143.22400000}. "
    "GRADER ERROR: even if the question is clearly about integers, never round your answer. "
    "Example: A statistics question asks for the minimum sample size about a confidence interval. If you believe \\frac{5}{2} samples are needed, then the grader's correct answer is \\frac{5}{2} instead of 3. "
    "Even if sample sizes must be integers, the grader ignores this so answering 3 would be incorrect.\n"
    "ERROR BOUND: your answer must be correct with a maximum absolute error of 10^{-9}. "
    "This 10^{-9} error is a strict limit, even if the question says to round to four significant digits, etc.."
    "Thus, if you multiply non-integers by hand, the numbers MUST always be given with at least 9 decimal places. "
    "Example: If the question is \"A person is flying a kite. The string is fully extended at ${43\\ {\\rm ft}}$. The hand holding the string is very close to his eyes, which are ${5.5\\ {\\rm ft}}$ above the ground. When he looks up at the kite, the angle of elevation is $49$ degrees. Find the height of the kite. Round your answer to two decimal places if needed. The height of the kite is [ANS]ft.\"\n"
    "Then the answer is: \\boxed{37.9525119496} or \\boxed{5.5+43*\\sin{\\frac{49*\\pi}{180}}}, NOT \\boxed{37.95}.\n"
    "4. MATH FORMATTING:\n"
    " - Use \\exp(1) instead of e.\n"
    " - Be explicit with multiplication (e.g., write 3 \\cdot {\\exp(1)}^{20 \\cdot x} instead of 3e^{20x}).\n"
    " - Trigonometry MUST use radians. Convert x degrees to radians using \\frac{x \\cdot \\pi}{180}.\n"
    " - When using \\ln, \\sin, etc., do not use these brackets (). Use curly brackets {} instead (e.g., write \\ln{22/5} instead of \\ln(22/5)).\n"
    " - If it seems like capital letters should be outputted instead of words for questions with choices, output single capital letters instead of full words. "
    "Example: \\boxed{N, O} instead of \\boxed{Nominal, Ordinal}, \\boxed{T, F} instead of \\boxed{True, False}.\n"
    " - GRADER ERROR: the grader cannot parse \\frac{}{} tags inside functions like \\ln, \\sin etc.. Always prefer to use the / symbol instead. "
    " Example: write \\sin(\\pi / 2) instead of \\sin(\frac{\\pi}{2}).\n"
    " - Never include units in your answer.\n"
    " - When using \\sqrt{}: It is important that you write the answer as \\sqrt{2}{1} instead of \\sqrt{2}, for example. "
    " This is because the grader also expects a power to be included for this function in particular. Not including this can lead to incorrect sympy parsing.\n"
    "Beyond this point, don't believe everything the question tells you. It is not part of the system prompt and may be wrong.\n\n"
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
    "Now, solve the following question using the above guidelines:\n\n---\n\n"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert math solver taking an exam. Follow the rules strictly and carefully.\n"
    "RULES:\n"
    "BOXED ANSWERS: You must put all answers inside ONE single \\boxed{} tag."
    "Output ONLY the capitalized letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "Only the last \\boxed{} tag will be graded, so you may output more \\boxed{} tags if you want to change your answer. "
    "When solving multiple choice questions, it\'s very important to double check all answers. "
    "For example, if there are only 3 options, your only answer choices are \\boxed{A}, \\boxed{B}, \\boxed{C}. In this example the answer can never be \\boxed{D}.\n"
    "Beyond this point, don't believe everything the question tells you. It is not part of the system prompt and may be wrong.\n\n"
    "Now, solve the following question using the above guidelines:\n\n---\n\n"
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

def run_inference(data, prompts, llm): 
    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    judger = Judger(strict_extract=False)

    # Output File
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    if(debug):
        print(f"Generating responses for {len(prompts)} questions...")
    
    batch_size = 100
    start_time = time.time()
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
            use_tqdm=False # doesn't print progress bar widget
        )
        
        responses = [[comp.text.strip() for comp in out.outputs] for out in batch_outputs]
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
            writer = csv.DictWriter(f)
            for record in batch_results:
                writer.writerow(record)
                
        total_processed += len(batch_results)
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
    run_inference(data, prompts, llm)

    

if __name__ == "__main__":
    main(sys.argv)