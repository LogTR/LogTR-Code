
import csv
import json
import argparse
import os
import re
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)
LOGHUB2_PATH = os.path.join(os.path.dirname(BASE_PATH), "loghub2DATA")


def convert_template_to_regex(pattern, wildcard_type='non_whitespace'):

    wildcard_map = {
        'non_whitespace': r'\S+',
        'any': r'.*',
        'any_lazy': r'.*?'
    }
    wildcard_regex = wildcard_map.get(wildcard_type, r'\S+')

    normalized_pattern = re.sub(r'<[^>]+>', '<*>', pattern)

    placeholder = "___WILDCARD___"
    temp_pattern = normalized_pattern.replace('<*>', placeholder)
    escaped = re.escape(temp_pattern)
    regex_pattern = escaped.replace(placeholder, wildcard_regex)
    return regex_pattern


def check_pattern_in_logs(csv_file_path, pattern, use_regex=False):
    
    if not os.path.exists(csv_file_path):
        return {
            'error': f"no such file: {csv_file_path}",
            'all_match': False,
            'total_count': 0
        }

    original_pattern = pattern
    auto_converted = False

    if '<*>' in pattern and not use_regex:
        converted_pattern = convert_template_to_regex(pattern, wildcard_type='any')
        print(f"    [Auto-convert] Detected <*> in pattern, auto-converting to regex")
        print(f"      Original pattern: {pattern}")
        print(f"      Regex pattern: {converted_pattern}")
        pattern = converted_pattern
        use_regex = True
        auto_converted = True

    total_count = 0
    match_count = 0
    mismatch_samples = []

    if use_regex:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return {
                'error': f"Regex error: {e}",
                'all_match': False,
                'total_count': 0
            }

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            content = row.get('Content', '')

            if use_regex:
                matched = regex.search(content) is not None
            else:
                matched = pattern in content

            if matched:
                match_count += 1
            else:
                if len(mismatch_samples) < 10:
                    mismatch_samples.append({
                        'LineId': row.get('LineId', ''),
                        'Content': content,
                        'EventId': row.get('EventId', '')
                    })

    mismatch_count = total_count - match_count
    match_rate = match_count / total_count if total_count > 0 else 0

    result = {
        'all_match': mismatch_count == 0,
        'total_count': total_count,
        'match_count': match_count,
        'mismatch_count': mismatch_count,
        'mismatch_samples': mismatch_samples,
        'match_rate': match_rate,
        'pattern': pattern,
        'use_regex': use_regex,
        'csv_file': csv_file_path
    }

    if auto_converted:
        result['original_pattern'] = original_pattern
        result['auto_converted'] = True

    return result


def check_pattern_by_event(system_name, event_id, pattern, use_regex=False):
    csv_file_path = os.path.join(LOGHUB2_PATH, system_name, f"{system_name}_{event_id}_logs.csv")

    if not os.path.exists(csv_file_path):
        print(f"[INFO] Log file not found, attempting to generate: {csv_file_path}")
        script_path = os.path.join(os.path.dirname(__file__), "get_all_you_want_log.py")
        if os.path.exists(script_path):
            import subprocess
            try:
                subprocess.run([
                    "python3", script_path,
                    "--system", system_name,
                    "--event_id", event_id
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                return {
                    'error': f"Failed to generate log file: {e.stderr}",
                    'all_match': False,
                    'total_count': 0
                }

    return check_pattern_in_logs(csv_file_path, pattern, use_regex)


def check_dual_patterns_with_sampling(system_name, event_id, new_pattern, old_pattern, sample_count=2):


    csv_file_path = os.path.join(LOGHUB2_PATH, system_name, f"{system_name}_{event_id}_logs.csv")

    if not os.path.exists(csv_file_path):
        return {'error': f"no such file: {csv_file_path}"}

    if '<*>' in new_pattern:
        new_regex = convert_template_to_regex(new_pattern, wildcard_type='any')
    else:
        new_regex = new_pattern

    if '<*>' in old_pattern:
        old_regex = convert_template_to_regex(old_pattern, wildcard_type='any')
    else:
        old_regex = old_pattern

    try:
        new_compiled = re.compile(new_regex)
        old_compiled = re.compile(old_regex)
    except re.error as e:
        return {'error': f"Regex error: {e}"}

    total_count = 0
    new_match_count = 0
    old_match_count = 0

    samples = {
        'new_match': [],
        'new_mismatch': [],
        'old_match': [],
        'old_mismatch': []
    }

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            content = row.get('Content', '')
            line_id = row.get('LineId', '')

            new_matched = new_compiled.search(content) is not None
            old_matched = old_compiled.search(content) is not None

            sample_item = {
                'LineId': line_id,
                'Content': content
            }

            if new_matched:
                new_match_count += 1
            if old_matched:
                old_match_count += 1

            if new_matched and len(samples['new_match']) < sample_count:
                samples['new_match'].append(sample_item)
            if not new_matched and len(samples['new_mismatch']) < sample_count:
                samples['new_mismatch'].append(sample_item)
            if old_matched and len(samples['old_match']) < sample_count:
                samples['old_match'].append(sample_item)
            if not old_matched and len(samples['old_mismatch']) < sample_count:
                samples['old_mismatch'].append(sample_item)

    new_match_rate = new_match_count / total_count if total_count > 0 else 0
    old_match_rate = old_match_count / total_count if total_count > 0 else 0

    return {
        'total_count': total_count,
        'new_match_rate': new_match_rate,
        'old_match_rate': old_match_rate,
        'samples': samples,
        'stats': {
            'new_match_count': new_match_count,
            'new_mismatch_count': total_count - new_match_count,
            'old_match_count': old_match_count,
            'old_mismatch_count': total_count - old_match_count
        },
        'patterns': {
            'new_pattern': new_pattern,
            'new_regex': new_regex,
            'old_pattern': old_pattern,
            'old_regex': old_regex
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check if all logs match a specific pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Check directly with CSV file
  python check_all_logs.py --csv /path/to/BGL_E255_logs.csv --pattern "error........"

  # Check by system name and EventId
  python check_all_logs.py --system BGL --event_id E255 --pattern "error........"

  # Use regular expression
  python check_all_logs.py --system BGL --event_id E255 --pattern "error\\.{8}" --regex
        """
    )

    parser.add_argument("--csv", type=str, help="CSV file path")
    parser.add_argument("--system", type=str, help="System name")
    parser.add_argument("--event_id", type=str, help="Event ID")

    parser.add_argument("--pattern", type=str, required=True, help="String pattern to match")
    parser.add_argument("--regex", action="store_true", help="Use regular expression matching")

    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode, only output JSON")

    args = parser.parse_args()

    if args.csv:
        result = check_pattern_in_logs(args.csv, args.pattern, args.regex)
    elif args.system and args.event_id:
        result = check_pattern_by_event(args.system, args.event_id, args.pattern, args.regex)
    else:
        parser.error("Must specify --csv or both --system and --event_id")

    if not args.quiet:
        print("=" * 80)
        print("Log Pattern Check Results")
        print("=" * 80)

        if 'error' in result:
            print(f"[Error] {result['error']}")
        else:
            print(f"File: {result['csv_file']}")
            print(f"Pattern: {result['pattern']}")
            print(f"Regex: {'Yes' if result.get('use_regex') else 'No'}")
            print("-" * 80)
            print(f"Total logs: {result['total_count']}")
            print(f"Matched: {result['match_count']}")
            print(f"Mismatched: {result['mismatch_count']}")
            print(f"Match rate: {result['match_rate']:.2%}")
            print("-" * 80)

            if result['all_match']:
                print("[Result] ✓ All logs match the pattern")
            else:
                print("[Result] ✗ Some logs don't match")
                if result['mismatch_samples']:
                    print("\nMismatch samples:")
                    for sample in result['mismatch_samples']:
                        print(f"  [LineId: {sample['LineId']}] {sample['Content']}")

        print("=" * 80)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")

    if args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0 if result.get('all_match', False) else 1


if __name__ == '__main__':
    exit(main())
