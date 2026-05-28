import json
import re
import sys
from tqdm import tqdm
from collections import Counter
from typing import List

### JSONL parsers

def extract_letter(text: str) -> str:
    """Extracts the MCQ letter."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def extract_boxed_answer(text: str) -> str:
    """Extracts the free-form math answer from \boxed{}, handling nested braces."""

    # Check for valid answer. If valid answer(s) exists, take the last instance. 
    box_str = "\\boxed{"
    start_idx = text.rfind(box_str) 
    if start_idx == -1: return ""


    # Find where the last brace ends. 
    content_start = start_idx + len(box_str)
    brace_count = 1
    for i in range(content_start, len(text)):
        if text[i] == '{': brace_count += 1
        elif text[i] == '}': brace_count -= 1
        
        if brace_count == 0: 
            return text[content_start:i].strip()
    return ""

def has_answer(s):
    if (s.rfind('</think>') == -1):
        return False
    else:
        s = s[s.rfind('</think>'):]
        if s.rfind('\\boxed{') == s.rfind('\\boxed{}'):
            return False
        
    return True

### Majority voting
def majority_vote_jsonl(file_paths: List[str], output_path: str, weights: list) -> None:
    """
    Performs majority voting across multiple JSONL result files.
    Aligns questions by their 'id' field.
    Tiebreaker: The file with the largest index in `file_paths` wins.

    Assumes all files contain the same set of question IDs.
    """
    
    if not file_paths:
        print("Error: No file paths provided.")
        return

    # Load all datasets into dictionaries keyed by id
    datasets = []
    for path in file_paths:
        file_dict = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    file_dict[item["id"]] = item
            datasets.append(file_dict)
        except Exception as e:
            print("error:", e)
    

    combined_results = []
    
    # Iterate through the IDs found in the first file
    base_ids = sorted(datasets[0].keys())
    tie_counts = [[] for _ in range(len(file_paths)+1)]

    for q_id in tqdm(base_ids):
        
        # Gather the items for this specific ID across all files
        items = [ds.get(q_id, {}) for ds in datasets]
        
        is_mcq = items[0].get("is_mcq", False)
        raw_responses = [item.get("response", "") for item in items]
        
        # Extract the answers
        extracted_answers = []
        answer_file_indices = []
        for idx, text in enumerate(raw_responses):
            if not has_answer(text):
                continue

            answer_file_indices.append(idx)
            if is_mcq:
                extracted_answers.append(extract_letter(text))
            else:
                extracted_answers.append(extract_boxed_answer(text))

        # No viable answer to choose from 
        if len(extracted_answers) == 0:
            for text in raw_responses:
                # Ensure the answer actually has some text, even if we know it's wrong
                if len(text) > 0:
                    new_record = {
                        "id": q_id,
                        "is_mcq": is_mcq,
                        "response": text
                    }
                    combined_results.append(new_record)
                    break

            tie_counts[0].append(q_id)  
            print("No answer:", q_id)
            continue

        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)

        match_counts = [weights[i] for i in answer_file_indices]
        for i in range(len(extracted_answers)-1):
            for j in range(i+1,len(extracted_answers)):
                gold = judger.split_by_comma(extracted_answers[j])
                result = judger.auto_judge(f'\\boxed{'{'}{extracted_answers[i]}{'}'}', gold, [chr(i+65) for i in range(26)])
                if result:
                    match_counts[i] += 1
                    match_counts[j] += 1

        # Count votes
        max_votes = max(match_counts)
        
        # Find tied votes
        tied_answers = [extracted_answers[i] for i in range(len(extracted_answers)) if match_counts[i] == max_votes]
        
        winning_answer = None
        winning_raw_text = ""
        
        # Resolve Ties
        if len(tied_answers) == 1:
            # Clear majority
            winning_answer = tied_answers[0]
            winning_index = extracted_answers.index(winning_answer)
            winning_raw_text = raw_responses[answer_file_indices[winning_index]]
        else:
            # Tiebreaker: largest index wins
            for file_idx in range(len(extracted_answers) - 1, -1, -1):
                ans = extracted_answers[file_idx]
                if ans in tied_answers:
                    winning_answer = ans
                    winning_raw_text = raw_responses[answer_file_indices[file_idx]]
                    break 
        
        tie_counts[max_votes].append(q_id)
        # Construct combined JSON object
        new_record = {
            "id": q_id,
            "is_mcq": is_mcq,
            "response": winning_raw_text
        }
        combined_results.append(new_record)

    # Write to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in combined_results:
            f.write(json.dumps(record) + "\n")
            
    print("Saved final results to:", output_path)
    print("Max vote counts:")
    for i in range(len(file_paths)+1):
        print(f'{i}:', tie_counts[i])

### Main

if __name__ == "__main__":
    
    input_files = [
        "results/voting/v1_initial.jsonl", 
        "results/voting/v1_reruns.jsonl", 
        "results/voting/v2.jsonl", 
        "results/voting/v3.jsonl", 
        "results/voting/v4.jsonl",
        "results/voting/v5.jsonl",
        "results/voting/v6.jsonl",
        "results/voting/v7.jsonl",
        "results/voting/v8.jsonl",
        "results/voting/v9.jsonl",
        "results/voting/v10.jsonl",
        "results/voting/v11.jsonl",
        "results/voting/v12.jsonl",
        "results/voting/lora_initial.jsonl"
    ]
    output_file = "results/FINAL_COMBINED.jsonl"

    weights = [1] * len(input_files)
    weights[-1] = 1
    
    majority_vote_jsonl(input_files, output_file, weights)