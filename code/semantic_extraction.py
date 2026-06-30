

import json
import os
import argparse
import random
from datetime import datetime
from tqdm import tqdm
from llm_client import LLMClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)

QWEN_API_KEY = ""
DEEPSEEK_API_KEY = ""
CLAUDE_API_KEY = ""
GEMINI_API_KEY = ""
GPT_API_KEY = ""
OLLAMA_API_KEY = ""


MODEL_NAME = "gpt-5.2"  #  qwen3-max, qwen-max-latest, deepseek-chat, deepseek-reasoner, claude-opus-4-5-20251101, claude-sonnet-4-20250514
TEMPERATURE = 0


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


FEW_SHOT_PATH = os.path.join(BASE_PATH, "golden_few_shots.json")


try:
    with open(FEW_SHOT_PATH, 'r', encoding='utf-8') as f:
        GOLDEN_EXAMPLES = json.load(f)
except FileNotFoundError:
    print("⚠️ golden_few_shots.json not found, will use default examples")
    GOLDEN_EXAMPLES = {}


def get_few_shot_examples(system_name):

    if system_name in GOLDEN_EXAMPLES:
        examples = GOLDEN_EXAMPLES[system_name]
    else:
        all_systems = list(GOLDEN_EXAMPLES.keys())
        if len(all_systems) >= 3:
            random_systems = random.sample(all_systems, 3)
            examples = [GOLDEN_EXAMPLES[sys][0] for sys in random_systems]
        else:
            examples = [
                {"log": "syslogd startup succeeded", "template": "<*> startup succeeded", "description": "The syslogd system log daemon started successfully."},
                {"log": "onStandStepChanged 3579", "template": "onStandStepChanged <*>", "description": "A change in standing step count was detected, with the current cumulative standing step count recorded as 3579."}
            ]

    example_text = ""
    for ex in examples:
        example_text += f"Input Log: {ex['log']}\n"
        example_text += f"Input Template: {ex['template']}\n"
        example_text += f"Output Description: {ex['description']}\n\n"

    return example_text.strip()


def build_description_prompt(system_name, log_content, template):
    """
    Build prompt for generating log descriptions - Dynamic Few-Shot version (V3.0)
    """

    few_shot_examples = get_few_shot_examples(system_name)

    prompt = f"""You are an expert system responsible for log data cleaning and event reconstruction. Your task is to translate raw system logs into accurate, objective natural language event descriptions.

**IMPORTANT: You must respond in English only.**

[Task Objective]
Convert the given log content into an "event statement". This statement must include all dynamic parameter values from the log (such as IPs, paths, numbers, IDs, etc.) and clearly explain what happened.

[Key Constraints]
1. **Do not explain terminology**: Do not explain what "{system_name}" or specific components are; directly describe what they did.
2. **Must include parameters**: Variable values in the log (corresponding to <*> in the template) are the core of the description and must be accurately included in the sentence.
3. **Objective statement tone**: Use declarative sentences, describing the events recorded in the log. For example: "Detected..." or "System executed...".
4. **Consistent format**: Output directly as one paragraph, without sections or titles.
5. **Language**: Output must be in English.

[Reference Examples ({system_name} or similar systems)]

{few_shot_examples}

---
[Task to Process]

System Name: {system_name}
Log Content: {log_content}
Log Template: {template}

Please output the event description for this log in English:"""

    return prompt


def generate_description(llm_client, system_name, log_content, template, max_retries=3):
    """
    Use LLM to generate description for a single log entry

    Args:
        llm_client: LLM client instance
        system_name: System name
        log_content: Log content
        template: Log template
        max_retries: Maximum retry attempts

    Returns:
        str: Generated description, or empty string if failed
    """
    prompt = build_description_prompt(system_name, log_content, template)

    for attempt in range(max_retries):
        try:
            response = llm_client.query(
                prompt=prompt,
                temperature=TEMPERATURE,
                system_prompt="You are an expert system for log interpretation and analysis. Please output the log event description directly, without any titles or prefixes."
            )
            return response.strip() if response else ""

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n  ⚠️  LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  ⏳  Waiting 3 seconds before retry...")
                import time
                time.sleep(3)
            else:
                print(f"\n  ❌ LLM call failed, maximum retries reached: {e}")
                return ""

    return ""


def process_dataset(dataset_path, output_path, llm_client, start_from=0, save_interval=10,
                   system_filter=None, max_samples_per_system=None):
    """
    Process the entire dataset, generating descriptions for each sample

    Args:
        dataset_path: Dataset JSON file path
        output_path: Output JSON file path
        llm_client: LLM client
        start_from: Start from which sample (for resuming)
        save_interval: Save after processing this many samples
        system_filter: Only process specified systems (list)
        max_samples_per_system: Maximum samples per system (for testing)
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if os.path.exists(output_path):
        print(f"📂 Found existing output file: {output_path}")
        print("📥 Loading for resume...")
        with open(output_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        print("📝 Creating new output file")
        output_data = {
            'generation_time': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'temperature': TEMPERATURE,
            'language': 'en',
            'total_systems': 0,
            'total_samples': 0,
            'systems': []  
        }

    existing_samples = {}
    if 'systems' in output_data:
        for sample in output_data['systems']:
            key = f"{sample.get('system_name', '')}_{sample.get('LineId', '')}"
            existing_samples[key] = sample

    total_samples = 0
    for system_name, system_data in dataset['systems'].items():
        if system_filter and system_name not in system_filter:
            continue
        samples = system_data.get('samples', [])
        if max_samples_per_system:
            samples = samples[:max_samples_per_system]
        total_samples += len(samples)

    print(f"\n📊 Dataset Overview")
    print(f"  Total systems: {len(dataset['systems'])}")
    print(f"  Total samples to process: {total_samples}")
    if system_filter:
        print(f"  Filtered systems: {system_filter}")
    if max_samples_per_system:
        print(f"  Max samples per system: {max_samples_per_system}")

    processed_count = 0
    skipped_count = 0
    system_count = 0

    for system_name, system_data in dataset['systems'].items():
        if system_filter and system_name not in system_filter:
            continue

        system_count += 1
        print(f"\n{'='*80}")
        print(f"🔧 Processing system: {system_name}")
        print(f"{'='*80}")

        samples = system_data.get('samples', [])
        if max_samples_per_system:
            samples = samples[:max_samples_per_system]
        for idx, sample in enumerate(tqdm(samples, desc=f"  {system_name}")):
            if processed_count < start_from:
                processed_count += 1
                skipped_count += 1
                continue

            line_id = sample.get('LineId', '')
            key = f"{system_name}_{line_id}"

            if key in existing_samples and existing_samples[key].get('description'):
                skipped_count += 1
                processed_count += 1
                continue

            description = generate_description(
                llm_client=llm_client,
                system_name=system_name,
                log_content=sample['log_content'],
                template=sample['EventTemplate']
            )

            output_sample = {
                'system_name': system_name,
                'EventId': sample['EventId'],
                'LineId': sample.get('LineId', ''),
                'template': sample['EventTemplate'],
                'log_content': sample['log_content'],
                'description': description
            }

            if key in existing_samples:
                for i, s in enumerate(output_data['systems']):
                    if s.get('system_name') == system_name and s.get('LineId') == line_id:
                        output_data['systems'][i] = output_sample
                        break
            else:
                output_data['systems'].append(output_sample)
                existing_samples[key] = output_sample

            processed_count += 1

            if processed_count % save_interval == 0:
                output_data['total_systems'] = system_count
                output_data['total_samples'] = len(output_data['systems'])

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"\n  💾 Auto-saved (processed: {processed_count})")

    output_data['total_systems'] = system_count
    output_data['total_samples'] = len(output_data['systems'])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Processing Complete!")
    print(f"{'='*80}")
    print(f"  Total processed: {processed_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Output saved to: {output_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Use LLM to generate descriptions for log samples")
    parser.add_argument("--input", type=str,
                       default="/evaluation_dataset.json",
                       help="Input dataset path")
    parser.add_argument("--output", type=str,
                       default=None,
                       help="Output file path (auto-generated if not specified)")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start from which sample (for resuming)")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save after processing this many samples")
    parser.add_argument("--systems", type=str, default=None,
                       help="Only process specified systems, comma-separated")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per system (for testing)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key (auto-selected based on model if not specified)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                       help="Model name: qwen3-max, qwen-max-latest, deepseek-chat, deepseek-reasoner, claude-opus-4-5-20251101, claude-sonnet-4-20250514")

    args = parser.parse_args()


    if args.output is None:
        model_tag = args.model.replace("-", "_").replace(".", "_")
        args.output = f"dataset_with_descriptions_{model_tag}.json"

    system_filter = None
    if args.systems:
        system_filter = [s.strip() for s in args.systems.split(',')]

    api_key = args.api_key if args.api_key else get_api_key(args.model)

    print(f"🚀 Initializing LLM client...")
    print(f"  Model: {args.model}")
    if args.model.lower().startswith('deepseek'):
        print(f"  Provider: DeepSeek")
    elif args.model.lower().startswith('claude'):
        print(f"  Provider: Claude")
    elif args.model.lower().startswith('gemini'):
        print(f"  Provider: Gemini")
    elif args.model.lower().startswith('gpt'):
        print(f"  Provider: GPT")
    elif args.model.lower().startswith('local'):
        print(f"  Provider: Ollama (Local)")
    else:
        print(f"  Provider: Qwen")
    print(f"  Temperature: {TEMPERATURE}")

    llm_client = LLMClient(
        model_type=args.model,
        api_key=api_key
    )

    process_dataset(
        dataset_path=args.input,
        output_path=args.output,
        llm_client=llm_client,
        start_from=args.start_from,
        save_interval=args.save_interval,
        system_filter=system_filter,
        max_samples_per_system=args.max_samples
    )


if __name__ == '__main__':
    main()
