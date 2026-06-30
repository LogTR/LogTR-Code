
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from llm_client import LLMClient


QWEN_API_KEY = ""
DEEPSEEK_API_KEY = ""
CLAUDE_API_KEY = ""
GEMINI_API_KEY = ""
GPT_API_KEY = ""
OLLAMA_API_KEY = ""


MODEL_NAME = "gpt-5.2"  #  qwen3-max, qwen-max-latest, deepseek-chat, deepseek-reasoner, claude-opus-4-5-20251101, claude-sonnet-4-20250514
TEMPERATURE = 0.1



def get_api_key(model_name):
    if model_name.lower().startswith("deepseek"):
        return DEEPSEEK_API_KEY
    elif model_name.lower().startswith("claude"):
        return CLAUDE_API_KEY
    elif model_name.lower().startswith("gemini"):
        return GEMINI_API_KEY
    elif model_name.lower().startswith("gpt"):
        return GPT_API_KEY
    elif model_name.lower().startswith("local"):
        return OLLAMA_API_KEY
    return QWEN_API_KEY


def normalize_text(text):

    if text is None:
        return ""
    return text.strip()


def load_few_shot_examples(few_shot_path):
    """
    Load few-shot examples

    Args:
        few_shot_path: Path to golden_few_shots.json file

    Returns:
        dict: {system_name: [examples]}
    """
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_task1_prompt(template, description, few_shot_examples=None):
    """
    Build Task 1 prompt (Optimized for Strict Format Adherence)
    """

    few_shot_text = ""
    if few_shot_examples:
        few_shot_text = "Reference Examples (Pay attention to the exact spacing):\n\n"
        for i, example in enumerate(few_shot_examples, 1):
            few_shot_text += f"Example {i}:\n"
            few_shot_text += f"Template: `{example['template']}`\n" 
            few_shot_text += f"Description: {example['description']}\n"
            few_shot_text += f"Log: `{example['log']}`\n\n" 
        few_shot_text += "---\n\n"

    prompt = f"""You are a precise log reconstruction engine. Your goal is to strictly reconstruct the original log based on the template.

{few_shot_text}Task Input:

Log Template:
```text
{template}
```

Event Description: {description}

CRITICAL INSTRUCTIONS:

Parameter Extraction: Identify the values for the <*> placeholders from the description.

Strict Template Adherence: You must preserve the template's structure EXACTLY as provided in the Log Template code block above.

Do NOT remove or add any spaces.

Do NOT correct "weird" spacing (e.g., if the template has ( <*> with a space, you MUST keep that space).

Do NOT change punctuation.

Output: Return ONLY the completed log text string. Do not wrap the output in markdown blocks or quotes.

Generated Log:"""
    return prompt


def evaluate_sample(llm_client, sample, system_name, few_shot_db=None, max_retries=3):
    """
    Evaluate a single sample

    Args:
        llm_client: LLM client
        sample: Sample data
        system_name: System name
        few_shot_db: Few-shot examples database (optional)
        max_retries: Maximum retry attempts

    Returns:
        dict: Evaluation result
    """
    template = sample['template']
    description = sample['description']
    ground_truth = sample['log_content']
    few_shot_examples = None
    if few_shot_db and system_name in few_shot_db:
        few_shot_examples = few_shot_db[system_name]
    prompt = build_task1_prompt(template, description, few_shot_examples)
    generated_log = None
    for attempt in range(max_retries):
        try:
            response = llm_client.query(
                prompt=prompt,
                temperature=TEMPERATURE,
                system_prompt="You are a professional log generation system. Please strictly follow the template and description to generate log text, outputting only the log text itself."
            )
            generated_log = normalize_text(response)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(3)
            else:
                generated_log = f"ERROR: {str(e)}"
    ground_truth_norm = normalize_text(ground_truth)
    exact_match = (generated_log == ground_truth_norm)

    return {
        'system_name': system_name,
        'EventId': sample['EventId'],
        'LineId': sample.get('LineId', ''),
        'template': template,
        'description': description,
        'ground_truth': ground_truth_norm,
        'generated_log': generated_log,
        'exact_match': exact_match
    }


def load_failed_samples(failed_path, system_filter=None):
    """
    Load failed samples from previous evaluation results

    Args:
        failed_path: Path to failed samples JSON file (e.g., *_failed.json)
        system_filter: Only load samples from specified systems (optional)

    Returns:
        list: List of (system_name, sample) tuples
    """
    with open(failed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_samples = []
    for sample in data.get('failed_samples', []):
        system_name = sample['system_name']
        if system_filter and system_name not in system_filter:
            continue
        eval_sample = {
            'system_name': system_name,
            'EventId': sample['EventId'],
            'LineId': sample.get('LineId', ''),
            'template': sample['template'],
            'description': sample['description'],
            'log_content': sample['ground_truth']  # ground_truth is the original log
        }
        all_samples.append((system_name, eval_sample))

    return all_samples


def run_evaluation(test_data_path, llm_client, output_path=None, system_filter=None, few_shot_path=None, failed_input=None):
    """
    Run Task 1 evaluation

    Args:
        test_data_path: Path to test data JSON file
        llm_client: LLM client
        output_path: Output result file path
        system_filter: Evaluate only specified systems (optional)
        few_shot_path: Few-shot examples file path (optional)
        failed_input: Path to failed samples JSON file for retry mode (optional)
    """
    few_shot_db = None
    if few_shot_path:
        print(f"Loading few-shot examples: {few_shot_path}")
        few_shot_db = load_few_shot_examples(few_shot_path)
        print(f"Loaded few-shot examples for {len(few_shot_db)} systems")
        for sys_name, examples in few_shot_db.items():
            print(f"  - {sys_name}: {len(examples)} examples")


    if failed_input:
        print(f"\n[RETRY MODE] Loading failed samples from: {failed_input}")
        all_samples = load_failed_samples(failed_input, system_filter)
        print(f"Loaded {len(all_samples)} failed samples for retry")
        if system_filter:
            print(f"  Filtered systems: {system_filter}")
    else:

        print(f"Loading test data: {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        all_samples = []
        for item in test_data.get('systems', []):
            system_name = item['system_name']
            if system_filter and system_name not in system_filter:
                continue
            all_samples.append((system_name, item))

    print(f"\nTotal samples to evaluate: {len(all_samples)}")


    results = {
        'metadata': {
            'task': 'task1_reconstruction',
            'test_time': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'temperature': TEMPERATURE,
            'test_data_source': failed_input if failed_input else test_data_path,
            'language': 'en',
            'retry_mode': failed_input is not None,
            'failed_input': failed_input
        },
        'samples': [],
        'summary': {}
    }


    print("\n" + "=" * 80)
    print("Starting Task 1 Evaluation: Directed Log Reconstruction")
    print("=" * 80)

    for system_name, item in tqdm(all_samples, desc="Evaluation Progress"):
        result = evaluate_sample(
            llm_client=llm_client,
            sample=item,
            system_name=system_name,
            few_shot_db=few_shot_db
        )
        results['samples'].append(result)


    total_samples = len(results['samples'])
    exact_match_count = sum(1 for r in results['samples'] if r['exact_match'])


    system_stats = {}
    for result in results['samples']:
        sys = result['system_name']
        if sys not in system_stats:
            system_stats[sys] = {'total': 0, 'exact_match': 0}
        system_stats[sys]['total'] += 1
        if result['exact_match']:
            system_stats[sys]['exact_match'] += 1

    results['summary'] = {
        'total_samples': total_samples,
        'exact_match_count': exact_match_count,
        'exact_match_rate': exact_match_count / total_samples if total_samples > 0 else 0,
        'by_system': system_stats
    }


    failed_output_path = None
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nComplete results saved to: {output_path}")

        failed_samples = [r for r in results['samples'] if not r['exact_match']]
        if failed_samples:
            failed_output_path = output_path.replace('.json', '_failed.json')
            failed_results = {
                'metadata': results['metadata'],
                'failed_count': len(failed_samples),
                'total_count': total_samples,
                'failed_samples': failed_samples
            }
            with open(failed_output_path, 'w', encoding='utf-8') as f:
                json.dump(failed_results, f, ensure_ascii=False, indent=2)
            print(f"Failed cases saved to: {failed_output_path}")

    print("\n" + "=" * 80)
    print("Task 1 Evaluation Summary")
    print("=" * 80)
    print(f"Total samples: {total_samples}")
    print(f"Exact matches: {exact_match_count} ({results['summary']['exact_match_rate']*100:.2f}%)")
    print(f"Failed: {total_samples - exact_match_count}")

    print("\n" + "-" * 80)
    print(f"{'System Name':<15s} {'Samples':>8s} {'Matches':>10s} {'Pass Rate':>10s}")
    print("-" * 80)

    for sys, stats in sorted(system_stats.items()):
        rate = stats['exact_match'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{sys:<15s} {stats['total']:>8d} {stats['exact_match']:>10d} {rate:>9.1f}%")

    print("=" * 80)

    return results, failed_output_path


def main():
    parser = argparse.ArgumentParser(description="Task 1 Evaluation: Directed Log Reconstruction")
    parser.add_argument("--input", type=str,
                       default=None,
                       help="Test data JSON file path (auto-selected based on model if not specified)")
    parser.add_argument("--output", type=str,
                       default=None,
                       help="Output result JSON file path")
    parser.add_argument("--few_shot", type=str,
                       default="/golden_few_shots.json",
                       help="Few-shot examples JSON file path")
    parser.add_argument("--zero_shot", action="store_true",
                       help="Use zero-shot mode (no few-shot examples)")
    parser.add_argument("--failed_input", type=str,
                       default=None,
                       help="Path to failed samples JSON file for retry mode (e.g., *_failed.json)")
    parser.add_argument("--api_key", type=str,
                       default=None,
                       help="API key (auto-selected based on model if not specified)")
    parser.add_argument("--model", type=str,
                       default=MODEL_NAME,
                       help="Model name: qwen3-max, qwen-max-latest, deepseek-chat, deepseek-reasoner, claude-opus-4-5-20251101, claude-sonnet-4-20250514")
    parser.add_argument("--temperature", type=float,
                       default=TEMPERATURE,
                       help="Temperature parameter")
    parser.add_argument("--systems", type=str,
                       default=None,
                       help="Evaluate only specified systems, comma-separated")

    args = parser.parse_args()

    if args.input is None:
        model_tag = args.model.replace("-", "_").replace(".", "_")
        args.input = f"descriptions_{model_tag}.json"
        print(f"Auto-selected input file based on model: {args.input}")

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ""
        os.makedirs(output_dir, exist_ok=True)


        model_tag = args.model.replace("-", "_").replace(".", "_")


        systems_tag = ""
        if args.systems:
            systems_tag = f"_{args.systems.replace(',', '_')}"

        if args.failed_input:
            args.output = f"{output_dir}/en_task1_retry_{model_tag}{systems_tag}_{timestamp}.json"
        else:
            args.output = f"{output_dir}/en_task1_results_{model_tag}{systems_tag}_{timestamp}.json"


    system_filter = None
    if args.systems:
        system_filter = [s.strip() for s in args.systems.split(',')]


    api_key = args.api_key if args.api_key else get_api_key(args.model)


    print(f"Initializing LLM client...")
    print(f"  Model: {args.model}")
    if args.model.lower().startswith('deepseek'):
        print(f"  Provider: DeepSeek")
    elif args.model.lower().startswith('claude'):
        print(f"  Provider: Claude")
    else:
        print(f"  Provider: Qwen")
    print(f"  Temperature: {args.temperature}")
    if args.failed_input:
        print(f"  Mode: RETRY (testing failed samples)")

    llm_client = LLMClient(
        model_type=args.model,
        api_key=api_key
    )


    few_shot_path = None if args.zero_shot else args.few_shot
    results, failed_output_path = run_evaluation(
        test_data_path=args.input,
        llm_client=llm_client,
        output_path=args.output,
        system_filter=system_filter,
        few_shot_path=few_shot_path,
        failed_input=args.failed_input
    )


if __name__ == '__main__':
    main()
