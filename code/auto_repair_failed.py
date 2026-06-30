
import json
import argparse
import os
import shutil
import subprocess
import re
from datetime import datetime
from collections import Counter
from llm_client import LLMClient
from check_all_logs import check_pattern_by_event, check_dual_patterns_with_sampling, convert_template_to_regex



QWEN_API_KEY = ""
DEEPSEEK_API_KEY = ""
CLAUDE_API_KEY = ""
GEMINI_API_KEY = ""
GPT_API_KEY = ""
OLLAMA_API_KEY = ""

MODEL_NAME = ""  #  qwen3-max, qwen-max-latest, deepseek-chat, deepseek-reasoner, claude-opus-4-5-20251101, claude-sonnet-4-20250514, gemini-3-pro-preview, gpt-5.2, local-*
TEMPERATURE = 0.0  


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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)
REPAIR_TEM_PATH = os.path.join(SCRIPT_DIR, "repair_tem")
LOGHUB2_PATH = os.path.join(os.path.dirname(BASE_PATH), "loghub2DATA")
FEW_SHOT_PATH = os.path.join(BASE_PATH, "golden_few_shots.json")

class DiagnosisResult:
    TEMPLATE_ERROR = "TEMPLATE_ERROR"
    DESCRIPTION_ERROR = "DESCRIPTION_ERROR"
    GENERATOR_ERROR = "GENERATOR_ERROR"  
    BOTH = "BOTH"
    NONE = "NONE"  


class RepairResult:
    CONTINUE = "CONTINUE"                        
    REDIRECT_TEMPLATE = "REDIRECT_TEMPLATE"      
    REDIRECT_DESCRIPTION = "REDIRECT_DESCRIPTION"  
    REDIRECT_GENERATOR = "REDIRECT_GENERATOR"    
    REDIRECT_SPLIT = "REDIRECT_SPLIT"            
    GIVE_UP = "GIVE_UP"                         


class RepairContext:
    def __init__(self, event_id, template, failed_samples, success_samples, diagnosis):
        self.event_id = event_id
        self.template = template
        self.failed_samples = failed_samples
        self.success_samples = success_samples
        self.diagnosis = diagnosis
        self.stage_history = []

    def add_stage_record(self, stage_name, llm_input, llm_output, conclusion, test_results=None):
        self.stage_history.append({
            'stage': stage_name,
            'input': llm_input,
            'output': llm_output,
            'conclusion': conclusion,
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        })

    def update_last_stage_test_results(self, test_results, conclusion):
        if self.stage_history:
            self.stage_history[-1]['test_results'] = test_results
            self.stage_history[-1]['conclusion'] = conclusion

    def build_history_context(self):
        if not self.stage_history:
            return ""

        context_parts = ["## Previous Analysis History\n"]

        for i, record in enumerate(self.stage_history, 1):
            input_text = record['input']
            if len(input_text) > 2000:
                input_text = input_text[:2000] + "\n...(truncated)"

            output_text = record['output']
            if len(output_text) > 1500:
                output_text = output_text[:1500] + "\n...(truncated)"

            context_parts.append(f"""
### Round {i}: {record['stage']}

**LLM Analysis Conclusion:**
{output_text}

**Test Results:** {record['conclusion']}
""")

        context_parts.append("\n---\n\nPlease continue the current stage based on the above history analysis. Avoid repeating previously attempted approaches.\n")
        return "\n".join(context_parts)

    def get_attempted_stages(self):
        return [record['stage'] for record in self.stage_history]


# ============================================================================
# Few-shot Examples
# ============================================================================

DIAGNOSIS_FEW_SHOTS = """
## Reference Cases

### Case 1
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E255",
  "LineId": "31861",
  "template": "d-cache flush parity error.......<*>",
  "description": "The BGL system detected a d-cache flush parity error, with the error count or status value recorded as 1.",
  "ground_truth": "d-cache flush parity error........1",
  "generated_log": "d-cache flush parity error.......1",
  "exact_match": false
}
```
Preliminary analysis: The generated_log does not match ground_truth. The difference is that "error.......1" is missing one ".". Looking at the template "error.......<*>", the number of "." matches generated_log but is less than ground_truth. This indicates generated_log follows the template, but the template may have an error. Need to retrieve more logs for EventId="E255" to confirm the number of ".".
```json
{
  "cause": "TEMPLATE_ERROR",
  "confidence": "MEDIUM",
  "analysis": "The difference between generated_log and ground_truth is the number of '.'. Template has 7 '.', while ground_truth has 8 '.'. Suspect one '.' was lost during template parsing",
  "template_issues": ["Number of '.' in template may be incorrect, need to retrieve more logs to confirm"],
  "description_issues": []
}
```

### Case 2
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E104",
  "LineId": "139903",
  "template": "ciod: for node <*> read continuation request but ioState is <*>",
  "description": "The BGL system's ciod component, while processing node 42, read a continuation request but found the ioState to be 0.",
  "ground_truth": "ciod: for node 42, read continuation request but ioState is 0",
  "generated_log": "ciod: for node 42 read continuation request but ioState is 0",
  "exact_match": false
}
```
Preliminary analysis: The generated_log does not match ground_truth. The difference is a missing "," after "42". Looking at the template "node <*> read", there is no ",", but ground_truth has ",". This indicates generated_log follows the template, but the template may have an error. Need to retrieve more logs for EventId="E104" to confirm if "," exists in all corresponding logs.
```json
{
  "cause": "TEMPLATE_ERROR",
  "confidence": "MEDIUM",
  "analysis": "The difference between generated_log and ground_truth is the comma. Template 'node <*> read' is missing a comma, while ground_truth is 'node 42, read'. Suspect comma was lost during template parsing",
  "template_issues": ["Template may be missing comma after 'node <*>'"],
  "description_issues": []
}
```

### Case 3
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E214",
  "LineId": "187386",
  "template": "correctable error detected in directory <*>",
  "description": "A correctable error was detected in directory 0, reporting a status state of 0.",
  "ground_truth": "correctable error detected in directory 0......0",
  "generated_log": "correctable error detected in directory 0",
  "exact_match": false
}
```
Preliminary analysis: The generated_log does not match ground_truth. The difference is that ground_truth ends with "directory 0......0" but generated_log ends with "directory 0". Looking at the description, both parameters 0 are mentioned. Looking at the template "correctable error detected in directory <*>", there is only one parameter position <*> after "directory". This may have led LLM to think only one parameter can be written, and the "......" cannot be inferred from template and description. I suspect "......" may be part of the template. Need to retrieve more logs for EventId="E214" to confirm if "......" exists in all corresponding logs.
```json
{
  "cause": "TEMPLATE_ERROR",
  "confidence": "MEDIUM",
  "analysis": "generated_log is missing '......0' part. Template has only one <*> placeholder but ground_truth has two parameters with '......' separator. Suspect template parsing is incomplete",
  "template_issues": ["Template may be missing '......<*>' part, insufficient placeholders"],
  "description_issues": []
}
```

### Case 4
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E252",
  "LineId": "188425",
  "template": "EndServiceAction <*> performed upon <*> by <*>",
  "description": "The system recorded that EndServiceAction 219 was performed on the component R33-M1-ND root.",
  "ground_truth": "EndServiceAction 219 performed upon R33-M1-ND by root",
  "generated_log": "EndServiceAction 219 performed upon root by R33-M1-ND",
  "exact_match": false
}
```
Preliminary analysis: The generated_log does not match ground_truth. The difference is that "R33-M1-ND by root" was generated as "root by R33-M1-ND". Both generated and ground_truth structures match the template structure, but parameter positions are wrong. Suspect the description has issues. The description "The system recorded that EndServiceAction 219 was performed on the component R33-M1-ND root." has ambiguous ending. It's difficult to determine the two parameter positions from the description. Suspect description needs optimization. Need to retrieve log context to understand correct parameter meanings.
```json
{
  "cause": "DESCRIPTION_ERROR",
  "confidence": "MEDIUM",
  "analysis": "generated_log structure matches template, but parameter order is wrong. Description 'R33-M1-ND root' is ambiguous, not clearly indicating which is the 'upon' object and which is the 'by' object",
  "template_issues": [],
  "description_issues": ["Description is not clear about parameter positions, 'R33-M1-ND root' causes ambiguity"]
}
```

### Case 5 (GENERATOR_ERROR - Generator sporadic error)
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E63",
  "LineId": "272704",
  "template": "<*> torus sender <*> retransmission error(s) (dcr <*>) detected and corrected",
  "description": "The system detected and corrected 1 retransmission error from the torus sender in the z+ direction, with the associated DCR register value of 0x02f8.",
  "ground_truth": "1 torus sender z+ retransmission error(s) (dcr 0x02f8) detected and corrected",
  "generated_log": "z+ torus sender 1 retransmission error(s) (dcr 0x02f8) detected and corrected",
  "exact_match": false
}
```
Successful sample with same template:
```json
{
  "LineId": "272702",
  "description": "The system detected and corrected 1 retransmission error from the torus sender in the y- direction, with the associated DCR register value of 0x02f7.",
  "ground_truth": "1 torus sender y- retransmission error(s) (dcr 0x02f7) detected and corrected",
  "generated_log": "1 torus sender y- retransmission error(s) (dcr 0x02f7) detected and corrected",
  "exact_match": true
}
```
Preliminary analysis: In the failed sample, generated_log does not match ground_truth. The difference is that the first parameter "1" and second parameter "z+" positions are swapped. However:
1. Checking template structure, `<*> torus sender <*>` matches ground_truth structure, template is correct
2. Checking failed sample description, it clearly mentions "1 retransmission error" and "z+ direction", information is complete and correct
3. Comparing successful sample description ("1 retransmission error from the torus sender in the y- direction"), the failed sample description structure and quality are consistent with successful sample, both clearly express error count and direction information
4. Key point: Same template has successful samples, and failed sample description quality is comparable to successful sample, proving both template and description are correct
5. Conclusion: This is a generator sporadic error, parameter order was incorrectly swapped
```json
{
  "cause": "GENERATOR_ERROR",
  "confidence": "HIGH",
  "analysis": "Same template has successful samples, indicating template structure is correct. Comparing successful and failed sample descriptions, both have consistent structure and quality, clearly expressing parameter information. Failed sample description is complete, but generator incorrectly swapped parameter order ('1' and 'z+' positions swapped). This is a generator sporadic error, regeneration can fix it.",
  "template_issues": [],
  "description_issues": []
}
```
"""

TEMPLATE_REPAIR_FEW_SHOTS = """
## Reference Cases

### Case 1: Fix missing punctuation (dot count)
**Step 1 - Preliminary Analysis:**
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E255",
  "LineId": "31861",
  "template": "d-cache flush parity error.......<*>",
  "description": "The BGL system detected a d-cache flush parity error, with the error count or status value recorded as 1.",
  "ground_truth": "d-cache flush parity error........1",
  "generated_log": "d-cache flush parity error.......1",
  "exact_match": false
}
```
Preliminary analysis conclusion: The difference between generated_log and ground_truth is the number of '.'. Template has 7 '.', while ground_truth has 8 '.'. Suspect one '.' was lost during template parsing. Determined as TEMPLATE_ERROR, need to retrieve more logs to confirm.

**Step 2 - Retrieval Verification:**
Retrieval results:
```
System: BGL
EventId: E255
Template: d-cache flush parity error.......<*>
Log count: 294
================================================================================
[LineId: 31861] d-cache flush parity error........1
[LineId: 155490] d-cache flush parity error........1
[LineId: 187335] d-cache flush parity error........0
[LineId: 205946] d-cache flush parity error........1
[LineId: 210822] d-cache flush parity error........1
[LineId: 218058] d-cache flush parity error........1
[LineId: 375603] d-cache flush parity error........1
[LineId: 375640] d-cache flush parity error........0
```
Further analysis: All retrieved logs show eight ".", but template has seven ".". Confirmed as template error. Recommend changing template from "d-cache flush parity error.......<*>" to "d-cache flush parity error........<*>". Found 294 related logs, need to set check_pattern to verify all logs.
```json
{
  "needs_repair": true,
  "old_template": "d-cache flush parity error.......<*>",
  "new_template": "d-cache flush parity error........<*>",
  "explanation": "Number of '.' in template is incorrect, should be 8 instead of 7, all retrieved logs show 8 '.'",
  "needs_check": true,
  "check_pattern": "error........",
  "check_pattern_is_regex": false,
  "confidence": "HIGH"
}
```

### Case 2: Fix missing punctuation (comma)
**Step 1 - Preliminary Analysis:**
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E104",
  "LineId": "139903",
  "template": "ciod: for node <*> read continuation request but ioState is <*>",
  "description": "The BGL system's ciod component, while processing node 42, read a continuation request but found the ioState to be 0.",
  "ground_truth": "ciod: for node 42, read continuation request but ioState is 0",
  "generated_log": "ciod: for node 42 read continuation request but ioState is 0",
  "exact_match": false
}
```
Preliminary analysis conclusion: The difference between generated_log and ground_truth is the comma. Template 'node <*> read' is missing a comma, while ground_truth is 'node 42, read'. Determined as TEMPLATE_ERROR, need to retrieve more logs to confirm.

**Step 2 - Retrieval Verification:**
Retrieval results:
```
System: BGL
EventId: E104
Template: ciod: for node <*> read continuation request but ioState is <*>
Log count: 395
================================================================================
[LineId: 139903] ciod: for node 42, read continuation request but ioState is 0
[LineId: 139927] ciod: for node 55, read continuation request but ioState is 0
[LineId: 139928] ciod: for node 50, read continuation request but ioState is 0
[LineId: 139929] ciod: for node 18, read continuation request but ioState is 0
[LineId: 139930] ciod: for node 25, read continuation request but ioState is 0
[LineId: 139931] ciod: for node 9, read continuation request but ioState is 0
```
Further analysis: All retrieved logs have "," before "read", but template doesn't have ",". Confirmed as template error. Recommend changing template from "ciod: for node <*> read" to "ciod: for node <*>, read". Need to set check_pattern to verify all logs.
```json
{
  "needs_repair": true,
  "old_template": "ciod: for node <*> read continuation request but ioState is <*>",
  "new_template": "ciod: for node <*>, read continuation request but ioState is <*>",
  "explanation": "Template is missing comma after 'node <*>', all retrieved logs show comma exists",
  "needs_check": true,
  "check_pattern": ", read continuation",
  "check_pattern_is_regex": false,
  "confidence": "HIGH"
}
```

### Case 3: Fix missing template structure (insufficient placeholders)
**Step 1 - Preliminary Analysis:**
Failed sample:
```json
{
  "system_name": "BGL",
  "EventId": "E214",
  "LineId": "187386",
  "template": "correctable error detected in directory <*>",
  "description": "A correctable error was detected in directory 0, reporting a status state of 0.",
  "ground_truth": "correctable error detected in directory 0......0",
  "generated_log": "correctable error detected in directory 0",
  "exact_match": false
}
```
Preliminary analysis conclusion: generated_log is missing '......0' part. Template has only one <*> placeholder but ground_truth has two parameters with '......' separator. Determined as TEMPLATE_ERROR, need to retrieve more logs to confirm template structure.

**Step 2 - Retrieval Verification:**
Retrieval results:
```
System: BGL
EventId: E214
Template: correctable error detected in directory <*>
Log count: 94
================================================================================
[LineId: 187386] correctable error detected in directory 0......0
[LineId: 187387] correctable error detected in directory 1......0
[LineId: 797723] correctable error detected in directory 0......0
[LineId: 797724] correctable error detected in directory 1......0
[LineId: 833543] correctable error detected in directory 0......0
[LineId: 833544] correctable error detected in directory 1......0
```
Further analysis: All retrieved logs have "......", but template doesn't have "......". Confirmed as template error. Recommend changing template from "correctable error detected in directory <*>" to "correctable error detected in directory <*>......<*>". Need to set check_pattern to verify all logs.
```json
{
  "needs_repair": true,
  "old_template": "correctable error detected in directory <*>",
  "new_template": "correctable error detected in directory <*>......<*>",
  "explanation": "Template is missing '......<*>' part, all retrieved logs show '......' separator and second parameter exist",
  "needs_check": true,
  "check_pattern": "......",
  "check_pattern_is_regex": false,
  "confidence": "HIGH"
}
```
"""

DESCRIPTION_REGEN_FEW_SHOTS = """
## Standard Examples of Good Descriptions

The following examples demonstrate the correct description style - semantic event statements, not copying log text verbatim:

### Example 1
Template: `instruction address: <*>`
Log: `instruction address: 0x0000df30`
Good description: The system recorded an instruction execution address of 0x0000df30.
Bad description: The log shows "instruction address: 0x0000df30".

### Example 2
Template: `Target=<*> Message=<*>`
Log: `Target=ido://...JTAG/8 Message=Pll failed to lock`
Good description: The system reported a PLL lock failure for target device ido://...JTAG/8. Note that the message uses "Pll" with only the first letter capitalized.
Bad description: The log content is "Target=ido://...JTAG/8 Message=Pll failed to lock".

### Example 3
Template: `wanted <*> got <*>`
Log: `wanted C X+ X- got C X+ Y-`
Good description: The system expected links C, X+, and X- but received C, X+, and Y-. Note that link identifiers are space-separated (not comma-separated).
Bad description: The exact log is "wanted C X+ X- got C X+ Y-" with space-separated values.

## Repair Case

### Case: Fix parameter order error caused by ambiguous description

**Problem Analysis:**
- Template: `EndServiceAction <*> performed upon <*> by <*>`
- Original description: "EndServiceAction 219 was performed on the component R33-M1-ND root."
- Problem: "R33-M1-ND root" in description is concatenated, cannot distinguish which is the operation target and which is the executor

**Repaired Description:**
NEW_DESCRIPTION: The system recorded that EndServiceAction 219 was performed upon the NodeCard component R33-M1-ND by the user root.

**Repair Key Points:**
1. Clarify parameter relationships: R33-M1-ND is the operation target, root is the executor
2. Use semantic description, don't copy log format
3. Maintain concise event statement style
"""

REDIRECT_DECISION_PROMPT = """
You are a log analysis expert. The current repair stage failed to solve the problem. Please determine the next action.

## Important Principles

**Never choose GIVE_UP unless absolutely necessary!**

- Only consider giving up after trying all possible directions
- If there are remaining redirects, prioritize trying other repair directions
- GIVE_UP means manual intervention is needed, should be the last choice

## Current Status

**Attempted stages**: {attempted_stages}
**Remaining redirects**: {remaining_redirects}
**Current stage**: {current_stage}
**Current stage conclusion**: {stage_conclusion}

{history_context}

## Failed Sample Information

{failed_samples_info}

## Your Choices

Please choose the next step based on the current situation. **Prioritize redirecting, unless there's truly no way forward**:

1. **REDIRECT_TEMPLATE**: Redirect to template repair stage (2.2a)
   - Applicable when: Suspect template structure has issues (punctuation, placeholder count, missing fixed text, etc.)
   - If template hasn't been carefully checked before, should redirect to try

2. **REDIRECT_DESCRIPTION**: Redirect to description repair stage (2.2b)
   - Applicable when: Template is correct, but description is inaccurate or ambiguous
   - If description optimization hasn't been tried before, should redirect to try

3. **REDIRECT_GENERATOR**: Redirect to generator retry stage (2.2c)
   - Applicable when: Both template and description are correct, may be generator sporadic error
   - If there are successful samples for reference, worth trying

4. **REDIRECT_SPLIT**: Redirect to template split analysis stage (2.2d)
   - Applicable when: Logs have multiple different structure/format variants, single template cannot cover all
   - Typical feature 1: Template can match log prefix, but some logs have extra suffix/extended content
   - Typical feature 2: **Parameter part contains variable number of repeating units** (e.g., multiple IDs connected by separators, ID count varies from 1 to N)
   - Example: Same EventId has both short and long logs, or parameter is a variable-length list
   - If LLM says "parameter count is variable, cannot represent with fixed <*>" in template repair stage, should choose this option

5. **GIVE_UP**: Give up repair (Last choice)
   - **Only when**: Multiple directions have been tried, and remaining redirects is 0 or 1
   - **Only when**: Confident the problem is beyond current system capability (e.g., data itself is wrong)
   - Must provide detailed reason for giving up and manual handling suggestions

## Decision Reference

- Remaining redirects >= 2: **Strongly recommend redirecting**, don't give up
- Remaining redirects == 1: **Still recommend redirecting**, unless very certain there's no solution
- Remaining redirects == 0: Can consider GIVE_UP, but must provide sufficient reason
- If logs have multiple format variants, prioritize considering REDIRECT_SPLIT

## Output Format (Strictly follow this JSON format)

```json
{{
    "decision": "<REDIRECT_TEMPLATE|REDIRECT_DESCRIPTION|REDIRECT_GENERATOR|REDIRECT_SPLIT|GIVE_UP>",
    "reason": "<Detailed explanation of why this decision was chosen, and why other options were not chosen>",
    "analysis": "<Comprehensive analysis based on historical context>",
    "final_diagnosis": "<If choosing GIVE_UP, provide final diagnosis type: UNKNOWN_ERROR/COMPLEX_PATTERN/DATA_QUALITY_ISSUE/OUT_OF_SCOPE, otherwise empty>",
    "suggestions": ["<If GIVE_UP, provide specific manual handling suggestions, otherwise empty array>"]
}}
```
"""

TEMPLATE_SPLIT_PROMPT = """
You are a log template analysis expert. The current template repair verification shows **partial match**, need to determine if it's "coarse template granularity" or "needs to split into multiple templates".

## Background

In log parsing, sometimes multiple different structured logs are incorrectly classified under one EventId. This may be because:
1. **Template granularity too coarse**: Template can be refined to uniformly match all logs
2. **Template needs splitting**: Logs essentially belong to different types, should be split into multiple templates

## Current Situation

**Original template**: `{old_template}`
**Suggested new template**: `{new_template}`

**Match Statistics**:
- New template match rate: {new_match_rate:.1%} ({new_match_count}/{total_count})
- Old template match rate: {old_match_rate:.1%} ({old_match_count}/{total_count})

## Sampled Logs

### Logs matched by new template
{new_match_samples}

### Logs not matched by new template
{new_mismatch_samples}

### Logs matched by old template
{old_match_samples}

### Logs not matched by old template
{old_mismatch_samples}

## Your Task

Analyze the above sampled logs and determine:

1. **REFINE**: If unmatched logs only have minor differences (like dot count, optional suffix, format variants), can adjust template to uniformly match
2. **SPLIT**: If unmatched logs have completely different structures, should split into multiple independent templates

## Output Format (Strictly follow this JSON format)

```json
{{
    "decision": "<REFINE|SPLIT>",
    "analysis": "<Detailed analysis of differences between two groups of logs>",
    "split_templates": [
        {{
            "template": "<Split template 1>",
            "check_pattern": "<Pattern for verification, can include <*>>",
            "description": "<Description of log types this template applies to>"
        }},
        {{
            "template": "<Split template 2>",
            "check_pattern": "<Pattern for verification>",
            "description": "<Description of log types this template applies to>"
        }}
    ],
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```

**Notes**:
- If choosing REFINE, split_templates can contain only one refined template
- If choosing SPLIT, split_templates must contain at least two templates
- **check_pattern must use `<*>` as variable placeholder**, regex syntax is prohibited
- **Prohibited**: `[0-9]`, `\d`, `.*`, `.+`, `\S+`, `\w+` and other regex expressions
- **Correct example**: `rts: kernel terminated for reason <*>` - use `<*>` to replace variables
- **Wrong example**: `rts: kernel terminated for reason [0-9]+` - Prohibited!
- **Do not add escape characters yourself**, write raw characters directly, system will handle automatically
- **Do not split words to build templates**, use `<*>` to replace complete variable parts
"""

TEMPLATE_SPLIT_FROM_LOGS_PROMPT = """
You are a log template analysis expert. The current EventId's logs have multiple different structures, please analyze and provide a solution.

## Current Template
`{template}`

## Log Length Analysis
{length_analysis}

## Grouped Log Samples

{group_samples}

## Failed Samples (generated results don't match expected)
{failed_samples}

## Your Task

Based on the above grouped logs, determine the log pattern type and provide corresponding solution:

1. **SPLIT**: Different groups have obviously different log structures (e.g., short logs only have basic info, long logs have extra fixed structures like parenthetical descriptions)
   -> Split into multiple independent templates

2. **REFINE**: Log structures are essentially the same, just some field lengths differ, can use one template to cover all
   -> Output a single refined template

3. **VARIABLE_LENGTH**: Parameter part is a variable-length list of repeating units (e.g., multiple IDs, multiple status units), essentially the same format
   -> Change corresponding `<*>` to `<*:list>` marker

4. **GIVE_UP**: Cannot determine or log structure is too complex

## Reference Cases

### Case 1: VARIABLE_LENGTH - Variable number of subcommand ID list

Original template: `Failed subcommands <*>`

Group samples:
- Group 1: `Failed subcommands 3503` (1 ID)
- Group 2: `Failed subcommands 3825\ 3845\ 3846` (3 IDs, separated by `\ `)
- Group 3: `Failed subcommands 3825\ 3845\ 3846\ 3847\ 3850` (5 IDs)

Analysis: All logs have the same fixed part (`Failed subcommands `), parameter is an ID list separated by `\ `, just different counts.
Conclusion: VARIABLE_LENGTH, new template `Failed subcommands <*:list>`

### Case 2: VARIABLE_LENGTH - Variable number of node status units

Original template: `inconsistent nodesets <*> <*> <ok> <*> <*> <ok>`

Group samples:
- Group 1: `inconsistent nodesets node-131 0x0000003e <ok> node-130 0x0000003e <ok>` (2 units)
- Group 2: `inconsistent nodesets node-131 0x0000003e <ok> node-130 0x0000003e <ok> node-129 0x0000003e <ok> node-128 0x0000003e <ok>` (4 units)

Analysis: Logs consist of repeating `node-name hex-value <ok>` units, variable count. Original template incorrectly used fixed number of `<*>` to match.
Conclusion: VARIABLE_LENGTH, new template `inconsistent nodesets <*:list>`

### Case 3: SPLIT - Two different log formats

Original template: `rts: kernel terminated for reason <*>`

Group samples:
- Group 1: `rts: kernel terminated for reason 1004` (only error code)
- Group 2: `rts: kernel terminated for reason 1004 (memory exhausted)` (has parenthetical description)

Analysis: Group 2 has structural elements that Group 1 doesn't have (parentheses and description text), this is structural difference, not list length difference.
Conclusion: SPLIT, split into two templates

### Case 4: REFINE - Unified coverage

Original template: `error code <*>`

Group samples:
- Group 1: `error code 404`
- Group 2: `error code 500123`

Analysis: Both are `error code ` followed by a number, just different number lengths, structure is completely the same.
Conclusion: REFINE, keep template `error code <*>`

## Output Format (Strictly follow this JSON format)

If SPLIT:
```json
{{
    "decision": "SPLIT",
    "analysis": "<Detailed analysis of structural differences between groups>",
    "split_templates": [
        {{"template": "<template 1>", "check_pattern": "<verification pattern>", "description": "<description>"}},
        {{"template": "<template 2>", "check_pattern": "<verification pattern>", "description": "<description>"}}
    ],
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```

If REFINE:
```json
{{
    "decision": "REFINE",
    "analysis": "<Detailed analysis>",
    "split_templates": [{{"template": "<refined template>", "check_pattern": "<verification pattern>", "description": "<description>"}}],
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```

If VARIABLE_LENGTH:
```json
{{
    "decision": "VARIABLE_LENGTH",
    "analysis": "<Detailed analysis of why it's variable length list, explain repeating unit structure>",
    "new_template": "<new template with variable length parameter changed to <*:list>>",
    "variable_description": "<describe format of variable length part, e.g.: ID list separated by backslash-space>",
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```

If GIVE_UP:
```json
{{
    "decision": "GIVE_UP",
    "analysis": "<reason cannot be handled>",
    "confidence": "LOW"
}}
```

**Notes**:
- If choosing SPLIT, split_templates must contain at least two templates
- If choosing REFINE, split_templates can contain only one refined template
- If choosing VARIABLE_LENGTH, change `<*>` at variable length parameter position to `<*:list>` in new_template
- **check_pattern must use `<*>` as variable placeholder**, regex syntax is prohibited
- **Prohibited**: `[0-9]`, `\d`, `.*`, `.+`, `\S+`, `\w+` and other regex expressions
- **Do not add escape characters yourself**, write raw characters directly, system will handle automatically
- When analyzing, focus on: log repeating patterns, fixed structure differences, whether parameter is essentially single value or list
"""

PATTERN_TYPE_FEW_SHOTS = """
## Reference Cases

### Case 1: VARIABLE_LENGTH - Variable number of subcommand ID list

**Original template**: `Failed subcommands <*>`

**Group log samples**:

Group 1 (parameter length < 10, about 45 entries):
- [LineId: 1001] Failed subcommands 3503
- [LineId: 1015] Failed subcommands 4201
- [LineId: 1089] Failed subcommands 5502

Group 2 (parameter length 10-30, about 28 entries):
- [LineId: 2001] Failed subcommands 3825\ 3845
- [LineId: 2034] Failed subcommands 4102\ 4103\ 4105
- [LineId: 2078] Failed subcommands 5501\ 5502

Group 3 (parameter length >= 30, about 12 entries):
- [LineId: 3001] Failed subcommands 3825\ 3845\ 3846\ 3847\ 3850
- [LineId: 3045] Failed subcommands 4101\ 4102\ 4103\ 4104\ 4105\ 4106
- [LineId: 3089] Failed subcommands 5501\ 5502\ 5503\ 5504

**Analysis**: Observing logs from each group:
1. All logs have the same fixed prefix: `Failed subcommands `
2. Parameter part is a numeric ID list, using `\ ` as separator
3. Short logs have 1 ID, medium logs have 2-3 IDs, long logs have 5-6 IDs
4. This is **the same log format**, just different list lengths, no need to split into multiple templates
5. Should mark `<*>` as variable length list `<*:list>`

```json
{
    "pattern_type": "VARIABLE_LENGTH",
    "analysis": "Parameter part is a numeric ID list separated by '\\\\ ', variable length (1-6 IDs), belongs to variable length list pattern",
    "new_template": "Failed subcommands <*:list>",
    "variable_description": "Subcommand ID list separated by backslash-space, variable count",
    "confidence": "HIGH"
}
```

### Case 2: VARIABLE_LENGTH - Variable number of node status units

**Original template**: `inconsistent nodesets <*>`

**Group log samples**:

Group 1 (parameter length < 50, about 30 entries):
- [LineId: 4001] inconsistent nodesets node-224 0x000001fe <ok>
- [LineId: 4023] inconsistent nodesets node-115 0x000000ff <ok>

Group 2 (parameter length 50-150, about 25 entries):
- [LineId: 5001] inconsistent nodesets node-224 0x000001fe <ok> 0x000001ff <fail> 0x00000200 <ok>
- [LineId: 5034] inconsistent nodesets node-115 0x000000ff <ok> 0x00000100 <fail>

Group 3 (parameter length >= 150, about 8 entries):
- [LineId: 6001] inconsistent nodesets node-224 0x000001fe <ok> 0x000001ff <fail> 0x00000200 <ok> 0x00000201 <fail> 0x00000202 <ok> 0x00000203 <ok>
- [LineId: 6045] inconsistent nodesets node-115 0x000000ff <ok> 0x00000100 <fail> 0x00000101 <ok> 0x00000102 <fail> 0x00000103 <ok>

**Analysis**: Observing logs from each group:
1. Fixed prefix: `inconsistent nodesets `
2. First is node identifier (e.g., `node-224`)
3. Followed by repeating pattern units: `address <status>`, e.g., `0x000001fe <ok>`
4. Short logs have 1 status unit, medium logs have 2-3, long logs have 5-6
5. This is the same format, just different number of status units, no need to split
6. Should mark as variable length

```json
{
    "pattern_type": "VARIABLE_LENGTH",
    "analysis": "Parameter part contains node ID and variable number of 'address <status>' units, belongs to variable length list pattern",
    "new_template": "inconsistent nodesets <*:list>",
    "variable_description": "Node identifier followed by variable number of 'hex-address <status>' units",
    "confidence": "HIGH"
}
```

### Case 3: SPLIT - Two different log formats (comparison example)

**Original template**: `rts: kernel terminated for reason <*>`

**Group log samples**:

Group 1 (parameter length < 10, about 40 entries):
- [LineId: 7001] rts: kernel terminated for reason 1004
- [LineId: 7023] rts: kernel terminated for reason 1008
- [LineId: 7056] rts: kernel terminated for reason 1012

Group 2 (parameter length >= 10, about 35 entries):
- [LineId: 8001] rts: kernel terminated for reason 1004 (memory exhausted, heap size exceeded)
- [LineId: 8034] rts: kernel terminated for reason 1008 (signal received: SIGTERM)
- [LineId: 8078] rts: kernel terminated for reason 1012 (I/O error on device sda)

**Analysis**: Observing logs from each group:
1. Group 1: Only error code, format is `rts: kernel terminated for reason error_code`
2. Group 2: Has error code and detailed description in parentheses, format is `rts: kernel terminated for reason error_code (detailed_description)`
3. These are **two different log formats**, not simple list length variation
4. Group 2 has structure that Group 1 doesn't have (parentheses and description text)
5. Should split into two templates

```json
{
    "pattern_type": "SPLIT",
    "analysis": "Two different formats exist: one with only error code, another with error code plus parenthetical description. This is structural difference, not list length difference",
    "split_reason": "Group 2 logs contain structural elements that Group 1 doesn't have (parentheses and description text)",
    "confidence": "HIGH"
}
```
"""

PATTERN_TYPE_JUDGMENT_PROMPT = """
You are a log template analysis expert. Logs under the current template are grouped by parameter length, please determine if this is "variable length list" or "different formats needing split".

## Background

In log parsing, when logs of the same EventId have large parameter length differences, it could be two situations:
1. **VARIABLE_LENGTH**: Parameter is a variable-length list (e.g., multiple IDs, multiple status units), essentially the same format
2. **SPLIT**: Structural differences exist, long logs contain fixed structural elements that short logs don't have (e.g., extra parentheses, keywords, etc.)

## Current Template
`{template}`

## Log Grouping Information
{length_analysis}

## Log Samples from Each Group (complete logs)
{group_samples}

{PATTERN_TYPE_FEW_SHOTS}

## Your Task

Carefully observe the **complete log samples** from each group and determine:

1. **VARIABLE_LENGTH**: If all logs have the same fixed parts, only the number of list elements at a certain parameter position differs
   - Feature: Parameter part has obvious repeating patterns or separators
   - Example: `item1\ item2\ item3` or `unit1 unit2 unit3`
   - When outputting new template, change variable length parameter position to `<*:list>`

2. **SPLIT**: If long logs contain structural elements that short logs don't have
   - Feature: Long logs have extra keywords, parentheses, or completely different suffix structures
   - Example: Short logs only have error code, long logs have `(detailed description)`

## Output Format

If VARIABLE_LENGTH:
```json
{{
    "pattern_type": "VARIABLE_LENGTH",
    "analysis": "<Detailed analysis of why it's variable length list>",
    "new_template": "<New template with variable length parameter changed to <*:list>>",
    "variable_description": "<Describe format of variable length part>",
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```

If SPLIT:
```json
{{
    "pattern_type": "SPLIT",
    "analysis": "<Detailed analysis of why split is needed>",
    "split_reason": "<Brief explanation of split reason>",
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```
"""


# ============================================================================
# Utility Functions
# ============================================================================

def load_failed_samples(failed_json_path):
    with open(failed_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('failed_samples', []), data.get('metadata', {})


def load_all_samples(result_json_path):
    with open(result_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_samples = data.get('results', []) or data.get('samples', [])
    failed_samples = [s for s in all_samples if not s.get('exact_match', True)]

    return all_samples, failed_samples, data.get('metadata', {})


# ============================================================================
# Repair Template Functions
# ============================================================================

def load_repair_template(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_repair_template(template_path, repair_template):
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(repair_template, f, ensure_ascii=False, indent=2)


def update_repair_template_entry(repair_template, event_id, updates):
    for template_entry in repair_template.get('templates', []):
        if template_entry.get('event_id') == event_id:
            template_entry.update(updates)
            return True
    return False


def update_repair_template_summary(repair_template):
    templates = repair_template.get('templates', [])
    completed = sum(1 for t in templates if t.get('status') == 'completed')
    failed = sum(1 for t in templates if t.get('status') == 'failed')
    pending = sum(1 for t in templates if t.get('status') == 'pending')
    in_progress = sum(1 for t in templates if t.get('status') == 'in_progress')

    repair_template['summary'].update({
        'templates_completed': completed,
        'templates_failed': failed,
        'templates_pending': pending,
        'overall_status': 'completed' if pending == 0 and in_progress == 0 else 'in_progress'
    })


def count_event_ids(failed_samples):
    counter = Counter(sample['EventId'] for sample in failed_samples)
    return counter.most_common()


def group_samples_by_event_id(failed_samples):
    groups = {}
    for sample in failed_samples:
        event_id = sample['EventId']
        if event_id not in groups:
            groups[event_id] = []
        groups[event_id].append(sample)
    return groups


def group_all_samples_by_event_id(all_samples):
    groups = {}
    for sample in all_samples:
        event_id = sample['EventId']
        if event_id not in groups:
            groups[event_id] = {
                'success': [],
                'failed': [],
                'template': sample.get('template', '')
            }

        if sample.get('exact_match', False):
            groups[event_id]['success'].append(sample)
        else:
            groups[event_id]['failed'].append(sample)

    return groups


def get_all_logs_for_event(system_name, event_id, output_dir=None):
    script_path = os.path.join(REPAIR_TEM_PATH, "get_all_you_want_log.py")
    if output_dir is None:
        output_dir = os.path.join(LOGHUB2_PATH, system_name)

    cmd = [
        "python3", script_path,
        "--system", system_name,
        "--event_id", event_id,
        "--output_dir", output_dir
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        json_file = os.path.join(output_dir, f"{system_name}_{event_id}_logs.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Failed to get logs: {e.stderr}")

    return None


def get_log_context(system_name, event_id, line_id, context=5, output_dir=None):
    script_path = os.path.join(REPAIR_TEM_PATH, "extract_log_context.py")
    if output_dir is None:
        output_dir = os.path.join(LOGHUB2_PATH, system_name)

    cmd = [
        "python3", script_path,
        "--system", system_name,
        "--event_id", event_id,
        "--line_id", str(line_id),
        "--context", str(context),
        "--output_dir", output_dir
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        output_file = os.path.join(
            output_dir,
            f"{system_name}_{event_id}_L{line_id}_C{context}.txt"
        )
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read()
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Failed to get context: {e.stderr}")

    return None


def load_few_shot_examples():
    if os.path.exists(FEW_SHOT_PATH):
        with open(FEW_SHOT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_task1_prompt(template, description, few_shot_examples=None):
    few_shot_text = ""
    if few_shot_examples:
        few_shot_text = "Reference examples (note the exact spacing):\n\n"
        for i, example in enumerate(few_shot_examples, 1):
            few_shot_text += f"Example {i}:\n"
            few_shot_text += f"Template: `{example['template']}`\n"
            few_shot_text += f"Description: {example['description']}\n"
            few_shot_text += f"Log: `{example['log']}`\n\n"
        few_shot_text += "---\n\n"

    prompt = f"""You are a precise log reconstruction engine. Your goal is to strictly reconstruct the original log according to the template.

{few_shot_text}Task Input:

Log Template:
```text
{template}
```

Event Description: {description}

Key Instructions:

Parameter Extraction: Identify values for <*> placeholders from the description.

Strictly Follow Template: You must generate exactly according to the template structure provided in the log template code block above.

Do not remove or add any spaces.

Do not "correct" strange spacing (e.g., if template has ( <*> with a space, you must keep that space).

Do not change punctuation.

Output: Only return the completed log text string. Do not wrap output with markdown blocks or quotes.

Generated log:"""
    return prompt


def test_single_sample(llm_client, template, description, ground_truth, system_name, few_shot_db=None, max_retries=3):
    few_shot_examples = None
    if few_shot_db and system_name in few_shot_db:
        few_shot_examples = few_shot_db[system_name]

    prompt = build_task1_prompt(template, description, few_shot_examples)

    generated = None
    for attempt in range(max_retries):
        try:
            response = llm_client.query(
                prompt=prompt,
                temperature=TEMPERATURE,
                system_prompt="You are a professional log generation system. Please strictly generate log text according to the template and description, only output the log text itself."
            )
            generated = response.strip() if response else ""
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(3)
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'generated': None,
                    'ground_truth': ground_truth.strip(),
                    'match': False
                }

    ground_truth_norm = ground_truth.strip()
    match = (generated == ground_truth_norm)

    return {
        'success': True,
        'generated': generated,
        'ground_truth': ground_truth_norm,
        'match': match
    }


# ============================================================================
# LLM Diagnosis and Repair
# ============================================================================

def ask_llm_for_redirect_decision(llm_client, current_stage, stage_conclusion,
                                   repair_context, remaining_redirects, failed_samples):
    attempted_stages = repair_context.get_attempted_stages()

    samples_info = []
    for sample in failed_samples[:5]:
        desc = sample.get('description', '')
        if len(desc) > 100:
            desc = desc[:100] + '...'
        samples_info.append(f"""- LineId: {sample.get('LineId')}
  Template: {sample.get('template')}
  Description: {desc}
  Expected: {sample.get('ground_truth')}
  Generated: {sample.get('generated_log', 'N/A')}""")

    history_context = repair_context.build_history_context()

    prompt = REDIRECT_DECISION_PROMPT.format(
        attempted_stages=", ".join(attempted_stages) if attempted_stages else "None",
        remaining_redirects=remaining_redirects,
        current_stage=current_stage,
        stage_conclusion=stage_conclusion,
        history_context=history_context,
        failed_samples_info="\n".join(samples_info)
    )

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a log analysis expert. Based on current repair status, decide the next action. Prioritize redirecting, don't give up unless absolutely necessary."
        )

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            result = json.loads(response)

        decision = result.get('decision', 'GIVE_UP').upper()

        if 'SPLIT' in decision:
            decision = RepairResult.REDIRECT_SPLIT
        elif 'TEMPLATE' in decision:
            decision = RepairResult.REDIRECT_TEMPLATE
        elif 'DESCRIPTION' in decision:
            decision = RepairResult.REDIRECT_DESCRIPTION
        elif 'GENERATOR' in decision:
            decision = RepairResult.REDIRECT_GENERATOR
        else:
            decision = RepairResult.GIVE_UP

        return {
            'decision': decision,
            'reason': result.get('reason', ''),
            'analysis': result.get('analysis', ''),
            'final_diagnosis': result.get('final_diagnosis', ''),
            'suggestions': result.get('suggestions', []),
            'raw_response': response
        }
    except Exception as e:
        print(f"  [Warning] Failed to parse redirect decision: {e}")
        if '2.2a' in current_stage and remaining_redirects > 0:
            return {
                'decision': RepairResult.REDIRECT_DESCRIPTION,
                'reason': f'LLM response parsing failed, defaulting to description repair',
                'analysis': '',
                'final_diagnosis': '',
                'suggestions': [],
                'error': str(e)
            }
        elif '2.2b' in current_stage and remaining_redirects > 0:
            return {
                'decision': RepairResult.REDIRECT_GENERATOR,
                'reason': f'LLM response parsing failed, defaulting to generator retry',
                'analysis': '',
                'final_diagnosis': '',
                'suggestions': [],
                'error': str(e)
            }
        else:
            return {
                'decision': RepairResult.GIVE_UP,
                'reason': f'LLM call failed: {str(e)}',
                'analysis': '',
                'final_diagnosis': 'UNKNOWN_ERROR',
                'suggestions': ['Manual inspection of LLM response needed'],
                'error': str(e)
            }


def diagnose_error_cause(llm_client, event_id, template, failed_samples, success_samples=None):
    failed_json_list = []
    for sample in failed_samples[:10]:
        sample_obj = {
            "system_name": sample.get('system_name', ''),
            "EventId": sample.get('EventId', ''),
            "LineId": sample.get('LineId', ''),
            "template": sample.get('template', ''),
            "description": sample.get('description', ''),
            "ground_truth": sample.get('ground_truth', ''),
            "generated_log": sample.get('generated_log', ''),
            "exact_match": sample.get('exact_match', False)
        }
        failed_json_list.append(json.dumps(sample_obj, ensure_ascii=False, indent=2))

    failed_samples_text = "\n\n".join(failed_json_list)

    success_section = ""
    if success_samples and len(success_samples) > 0:
        success_json_list = []
        for sample in success_samples[:5]:
            sample_obj = {
                "LineId": sample.get('LineId', ''),
                "description": sample.get('description', ''),
                "ground_truth": sample.get('ground_truth', ''),
                "generated_log": sample.get('generated_log', ''),
                "exact_match": True
            }
            success_json_list.append(json.dumps(sample_obj, ensure_ascii=False, indent=2))
        success_section = f"""
## Successful Samples with Same Template (total {len(success_samples)})

The following are samples successfully reconstructed using the same template, for reference:

{chr(10).join(success_json_list)}
"""

    diagnosis_input = f"""## Test Results for EventId: {event_id}
Template: {template}
Successful samples: {len(success_samples) if success_samples else 0}
Failed samples: {len(failed_samples)}
{success_section}
## Failed Samples (total {len(failed_samples)})

{failed_samples_text}"""

    prompt = f"""You are a log analysis expert. I need you to analyze why the LLM failed to correctly reconstruct these logs.

## Background

1. **Template Source**: These templates were automatically extracted from original logs by log parsing tools (like Drain). They may contain parsing errors such as missing punctuation, incorrect parameter counts, or incomplete patterns.

2. **Description Source**: These descriptions were generated by another LLM based on log content and templates, intended to explain the meaning of logs. They may be inaccurate, missing key parameter values, or not detailed enough to reconstruct the exact log.

3. **Task Background**: We asked the LLM to reconstruct the original log (ground_truth) based on template and description, but it generated a different result (generated_log) that doesn't match the original.

Your task: Analyze whether the mismatch is caused by template issues, description issues, both, or undeterminable causes.

## Important Judgment Principles

1. **Prioritize Template Repair**: Template repair is "fix once, benefit all", while description repair needs to be done one by one. When both approaches are viable, should prioritize determining as TEMPLATE_ERROR.

2. **Template Granularity Issues**: If parameter positions in ground_truth have fixed prefix/suffix structures (like `mLctn()`, `prefix=`, `Type()`, etc.), and these fixed texts are encompassed by `<*>` in current template, this indicates template granularity is too coarse, should be determined as TEMPLATE_ERROR, not expecting description to supplement these format information.

3. **Sampling Rule Hint**: We sample at most 3 logs per template for testing. If multiple samples (especially all 3) of the same template fail, this strongly suggests template itself has issues, not individual description problems. When seeing multiple same-template failures, think more actively about template optimization approaches.

4. **Fixed Pattern Recognition**: When all failed samples' ground_truth have the same fixed text prefix or suffix at a certain `<*>` position, and generated_log is missing these fixed texts, should first consider incorporating these fixed texts into template, rather than requiring description to provide this format information.

5. **Generator Sporadic Error Recognition**: If same template has successful samples, it indicates template and description structure itself may not have issues. Should compare description quality between successful and failed samples:
   - If failed sample descriptions have similar structure and quality as successful samples, both clearly expressing parameter information, then description is also fine
   - In this case, failure is likely generator's sporadic error (like parameter order swap, random format detail errors, etc.)
   - Should determine as GENERATOR_ERROR, regeneration can fix it

## Key Field Descriptions
- template: Log template with <*> as parameter placeholders
- description: Natural language description of the log
- ground_truth: The actual original log we expect to reconstruct
- generated_log: What LLM actually generated (incorrect)
- exact_match: false means generated_log doesn't match ground_truth

{DIAGNOSIS_FEW_SHOTS}

{diagnosis_input}

## Your Analysis (Chain of Thought)

Please analyze step by step:

**Step 1**: Compare generated_log and ground_truth, find the exact differences (missing characters, extra characters, wrong values, etc.)

**Step 2**: Check if template structure matches ground_truth. This is a key step, need to carefully check:
- Missing punctuation in template (e.g., periods, commas, quotes)
- Wrong number of `<*>` placeholders
- Missing fixed text patterns
- **Key point**: Check if each `<*>` position in ground_truth has fixed prefix/suffix (like `name=`, `Type()`, `id:`, etc.). If yes, but template doesn't include these fixed texts, then template granularity is too coarse, should determine as TEMPLATE_ERROR
- **Key point**: Even if `<*>` count looks correct, check if fixed text was incorrectly included in parameter range

**Step 3**: Check if description contains all information needed to correctly fill <*> placeholders.

**Step 4**: Determine root cause based on your analysis. Remember: If problem can be solved by refining template, prioritize determining as TEMPLATE_ERROR.

## Output Format (Strictly follow this JSON format):

```json
{{
    "cause": "<TEMPLATE_ERROR|DESCRIPTION_ERROR|GENERATOR_ERROR|BOTH|NONE>",
    "confidence": "<HIGH|MEDIUM|LOW>",
    "analysis": "<Your detailed step-by-step analysis>",
    "template_issues": ["<List of specific template issues found, empty if none>"],
    "description_issues": ["<List of specific description issues found, empty if none>"]
}}
```

Where:
- **TEMPLATE_ERROR**: Template itself is incorrect (wrong punctuation, wrong placeholder count, missing patterns, granularity too coarse causing fixed text to be included in parameters)
- **DESCRIPTION_ERROR**: Description missing information or has wrong values
- **GENERATOR_ERROR**: Both template and description are correct, but generator has sporadic errors (like parameter order swap). Usually occurs when same template has successful samples
- **BOTH**: Both template and description have issues
- **NONE**: Cannot determine issue or failure is due to other reasons
"""

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a professional log analysis expert. Analyze error patterns and determine root cause of log reconstruction failures. Always output in the specified JSON format."
        )

        cause = DiagnosisResult.NONE
        analysis = response
        confidence = "LOW"
        template_issues = []
        description_issues = []

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

            result_json = json.loads(json_str)
            cause_text = result_json.get('cause', '').upper()

            if cause_text == 'TEMPLATE_ERROR':
                cause = DiagnosisResult.TEMPLATE_ERROR
            elif cause_text == 'DESCRIPTION_ERROR':
                cause = DiagnosisResult.DESCRIPTION_ERROR
            elif cause_text == 'GENERATOR_ERROR':
                cause = DiagnosisResult.GENERATOR_ERROR
            elif cause_text == 'BOTH':
                cause = DiagnosisResult.BOTH
            elif cause_text == 'NONE':
                cause = DiagnosisResult.NONE

            confidence = result_json.get('confidence', 'LOW').upper()
            analysis = result_json.get('analysis', response)
            template_issues = result_json.get('template_issues', [])
            description_issues = result_json.get('description_issues', [])

        except (json.JSONDecodeError, AttributeError):
            if "CAUSE:" in response:
                lines = response.split('\n')
                for line in lines:
                    if line.strip().startswith("CAUSE:"):
                        cause_text = line.replace("CAUSE:", "").strip().upper()
                        if "GENERATOR" in cause_text:
                            cause = DiagnosisResult.GENERATOR_ERROR
                        elif "TEMPLATE" in cause_text and "DESCRIPTION" in cause_text:
                            cause = DiagnosisResult.BOTH
                        elif "TEMPLATE" in cause_text:
                            cause = DiagnosisResult.TEMPLATE_ERROR
                        elif "DESCRIPTION" in cause_text:
                            cause = DiagnosisResult.DESCRIPTION_ERROR
                        elif "NONE" in cause_text:
                            cause = DiagnosisResult.NONE

        return {
            'cause': cause,
            'confidence': confidence,
            'analysis': analysis,
            'template_issues': template_issues,
            'description_issues': description_issues,
            'raw_response': response,
            'diagnosis_input': diagnosis_input,
            'diagnosis_output': response
        }
    except Exception as e:
        return {
            'cause': DiagnosisResult.NONE,
            'confidence': 'LOW',
            'analysis': str(e),
            'template_issues': [],
            'description_issues': [],
            'raw_response': None,
            'diagnosis_input': diagnosis_input if 'diagnosis_input' in dir() else '',
            'diagnosis_output': '',
            'error': str(e)
        }


def suggest_template_repair(llm_client, system_name, event_id, old_template, failed_samples, all_logs_data, diagnosis_context=None, repair_context=None):
    all_logs = all_logs_data.get('logs', [])
    total_count = len(all_logs)

    if total_count <= 50:
        sampled_logs = all_logs
    else:
        front_count = 20
        mid_count = 15
        back_count = 15

        front_logs = all_logs[:front_count]
        mid_start = (total_count - mid_count) // 2
        mid_logs = all_logs[mid_start:mid_start + mid_count]
        back_logs = all_logs[-back_count:]

        seen_line_ids = set()
        sampled_logs = []
        for log in front_logs + mid_logs + back_logs:
            if log['LineId'] not in seen_line_ids:
                seen_line_ids.add(log['LineId'])
                sampled_logs.append(log)

    logs_text = "\n".join([f"[LineId: {log['LineId']}] {log['Content']}" for log in sampled_logs])

    context_section = ""
    if diagnosis_context:
        context_section = f"""
## Step 1 - Preliminary Analysis (Completed)

**Input failed samples:**
{diagnosis_context.get('diagnosis_input', '')}

**Preliminary analysis conclusion:**
{diagnosis_context.get('diagnosis_output', '')}

---

## Step 2 - Retrieval Verification (Current Stage)
"""

    history_context = ""
    if repair_context:
        history_context = repair_context.build_history_context()

    prompt = f"""You are a log template expert. This is a two-stage analysis task, you are now at step 2.

{TEMPLATE_REPAIR_FEW_SHOTS}

{context_section}

{history_context}

Based on preliminary analysis, the template may have issues. We have retrieved actual log instances for this EventId from the original log file.

**Important**: The logs shown below are only partial samples. Total log count is {all_logs_data.get('actual_count', 'unknown')}. Before finalizing template repair, we may need to verify all logs match the suggested pattern.

## Current Template Information

System: {system_name}
EventId: {event_id}
Current template: `{old_template}`
Total logs for this template: {all_logs_data.get('actual_count', 'unknown')}
Showing {len(sampled_logs)} sample logs below (mixed sampling: front + middle + end)

## Retrieved Actual Log Samples

{logs_text}

## Your Task

Based on step 1 preliminary analysis and step 2 retrieved actual logs:
1. Verify if preliminary analysis guess is correct
2. Check actual logs to understand the real pattern
3. Compare with current template
4. **Determine the real error cause**:
   - TEMPLATE_ERROR: Template structure has issues (punctuation, placeholders, fixed text, etc.)
   - DESCRIPTION_ERROR: Template is correct, but description is inaccurate or ambiguous (e.g., case sensitivity, parameter value range, etc.)
   - GENERATOR_ERROR: Both template and description are correct, may be generator sporadic error
   - NONE: No repair needed
5. If template repair is needed, specify a pattern to verify against all logs

## Important Principle: Minimal Changes

1. **Conservative repair**: Only fix clear errors, don't change template structure just for "unified matching"
2. **Keep original template skeleton**: Prioritize fixing punctuation, spaces and other details, avoid changing fixed text to wildcards
3. **Keep original when uncertain**: If sample logs have multiple variants that are hard to unify, keep original template structure for system to verify later

## Output Format (Strictly follow this JSON format):

```json
{{
    "needs_repair": true/false,
    "confirmed_cause": "<TEMPLATE_ERROR|DESCRIPTION_ERROR|GENERATOR_ERROR|NONE>",
    "old_template": "{old_template}",
    "new_template": "<corrected template, same as old if no repair needed>",
    "explanation": "<detailed explanation of the issue and how you fixed it, or explain why it's determined to be other type of error>",
    "needs_check": true/false,
    "check_pattern": "<substring or regex pattern that all logs should contain to verify the fix>",
    "check_pattern_is_regex": false,
    "confidence": "<HIGH|MEDIUM|LOW>"
}}
```

**confirmed_cause Guide**:
- If you find template structure indeed has issues, set to TEMPLATE_ERROR
- If template structure is correct but description has errors (e.g., parameter case, value range, etc.), set to DESCRIPTION_ERROR
- If both template and description are correct, just generator sporadic error, set to GENERATOR_ERROR
- If cannot determine or no repair needed, set to NONE

**check_pattern Guide**:
- check_pattern must be a **single substring** or **regex**, cannot be a simple list of multiple patterns
- If you change punctuation (e.g., periods), set check_pattern to the exact sequence (e.g., "error........")
- **Can directly use `<*>` as variable placeholder**, system will automatically convert to regex match
  - Example: verify `symbol <*>, bit` structure -> directly write `symbol <*>, bit`
  - Example: verify leading number format -> directly write `<*> ddr errors`
- **Important: When using `<*>`, do not manually escape any characters!**
  - Write raw characters directly, system will handle escaping automatically
  - Correct example: `mLctn(<*>), mCardSernum(<*>)` <- write parentheses directly, don't escape
  - Wrong example: `mLctn\\(<*>\\)` <- don't add backslashes yourself!
- If not using `<*>` but need pure regex, set `check_pattern_is_regex: true`, then manual escaping is needed
- **Wrong examples**:
  - `", symbol , bit "` <- Wrong! Variable position cannot be empty, should write `symbol <*>, bit`
- Set needs_check to true when total logs > displayed samples AND you are making structural changes
"""

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a log template expert. Carefully analyze log patterns and suggest accurate template corrections. Always output in the specified JSON format."
        )

        if repair_context:
            repair_context.add_stage_record(
                stage_name='2.2a-TemplateRepair',
                llm_input=prompt,
                llm_output=response,
                conclusion='Pending test verification'
            )

        result = {
            'needs_repair': False,
            'confirmed_cause': 'TEMPLATE_ERROR',
            'old_template': old_template,
            'new_template': old_template,
            'explanation': '',
            'needs_check': False,
            'check_pattern': '',
            'check_pattern_is_regex': False,
            'confidence': 'LOW',
            'raw_response': response
        }

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

            result_json = json.loads(json_str)

            result['needs_repair'] = result_json.get('needs_repair', False)
            result['confirmed_cause'] = result_json.get('confirmed_cause', 'TEMPLATE_ERROR').upper()
            result['new_template'] = result_json.get('new_template', old_template)
            result['explanation'] = result_json.get('explanation', '')
            result['needs_check'] = result_json.get('needs_check', False)
            result['check_pattern'] = result_json.get('check_pattern', '')
            result['check_pattern_is_regex'] = result_json.get('check_pattern_is_regex', False)
            result['confidence'] = result_json.get('confidence', 'LOW').upper()

        except (json.JSONDecodeError, AttributeError):
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("NEEDS_REPAIR:"):
                    result['needs_repair'] = "YES" in line.upper()
                elif line.startswith("NEW_TEMPLATE:"):
                    result['new_template'] = line.replace("NEW_TEMPLATE:", "").strip()
                elif line.startswith("EXPLANATION:"):
                    result['explanation'] = line.replace("EXPLANATION:", "").strip()

        return result

    except Exception as e:
        return {
            'needs_repair': False,
            'confirmed_cause': 'NONE',
            'old_template': old_template,
            'new_template': old_template,
            'explanation': str(e),
            'needs_check': False,
            'check_pattern': '',
            'check_pattern_is_regex': False,
            'confidence': 'LOW',
            'raw_response': None,
            'error': str(e)
        }


def regenerate_description(llm_client, system_name, event_id, line_id, template, log_content, old_description, context_text, diagnosis_context=None, repair_context=None):
    sample_json = json.dumps({
        "system_name": system_name,
        "EventId": event_id,
        "LineId": line_id,
        "template": template,
        "log_content": log_content,
        "old_description": old_description
    }, ensure_ascii=False, indent=2)

    diagnosis_section = ""
    if diagnosis_context:
        diagnosis_section = f"""
## Step 1 - Preliminary Analysis (Completed)

**Input failed samples:**
{diagnosis_context.get('diagnosis_input', '')}

**Preliminary analysis conclusion:**
{diagnosis_context.get('diagnosis_output', '')}

---

## Step 2 - Retrieve Context and Fix Description (Current Stage)
"""

    history_context = ""
    if repair_context:
        history_context = repair_context.build_history_context()

    prompt = f"""You are a log description expert. This is a two-stage analysis task, you are now at step 2.

{DESCRIPTION_REGEN_FEW_SHOTS}

{diagnosis_section}

{history_context}

Based on preliminary analysis, the current description failed to help LLM correctly reconstruct the original log. We have retrieved context information for this log.

## Current Failed Sample

{sample_json}

## Retrieved Log Context (surrounding logs for reference)

{context_text}

## Your Task

Based on step 1 preliminary analysis and step 2 retrieved context:
1. Analyze log content and its context
2. Understand what information is missing or wrong in the old description
3. Generate a new, more accurate description

## Requirements for New Description

**Core Principle: Description is a semantic event statement, like an alert message from a monitoring system, not a verbatim copy of the log.**

### Rules that must be followed:

1. **Do not copy log verbatim**: Don't write "The log shows..." or "The exact content is..." that directly quote the log
2. **Use event statement tone**: Use semantic expressions like "The system reported...", "An error occurred...", "The component detected..."
3. **Parameter values must be included**: Variable values in the log (corresponding to <*> in template) must be accurately included in the sentence
4. **Clarify parameter relationships**: Clearly explain relationships and order between parameters, avoid ambiguity

### Parameter Internal Format Notes (allowed):

When parameter values have special format requirements (like case, separators) that affect reconstruction, can use parentheses or "Note that..." to explain:
- Good: "...reported a PLL lock failure. Note that the message uses 'Pll' with only the first letter capitalized."
- Good: "...expected links C, X+, and X-. Note that identifiers are space-separated, not comma-separated."
- Bad: "The log content is 'Message=Pll failed to lock'."

### Description Style Comparison:

- Good: "The system recorded an instruction execution address of 0x0000df30."
- Bad: "The log shows instruction address: 0x0000df30."

- Good: "A floating-point exception occurred with type SNAN and value 0."
- Bad: "The exact log content is 'invalid (SNAN)...0'."

## Your Output

Generate a semantic event description that enables LLM to accurately reconstruct the log.

Output format:
NEW_DESCRIPTION: <your improved description>
"""

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a log description expert. Generate precise descriptions that enable exact log reconstruction from templates."
        )

        if repair_context:
            repair_context.add_stage_record(
                stage_name='2.2b-DescriptionRepair',
                llm_input=prompt,
                llm_output=response,
                conclusion='Pending test verification'
            )

        new_description = response
        if "NEW_DESCRIPTION:" in response:
            new_description = response.split("NEW_DESCRIPTION:")[-1].strip()

        return {
            'old_description': old_description,
            'new_description': new_description,
            'raw_response': response
        }
    except Exception as e:
        return {
            'old_description': old_description,
            'new_description': old_description,
            'error': str(e),
            'raw_response': None
        }


def analyze_template_split(llm_client, system_name, event_id, old_template, new_template,
                           sampling_result, repair_context=None):
    def format_samples(samples, label):
        if not samples:
            return f"(No {label} samples)"
        return "\n".join([f"- [LineId: {s['LineId']}] {s['Content']}" for s in samples])

    new_match_samples = format_samples(sampling_result['samples']['new_match'], 'new template match success')
    new_mismatch_samples = format_samples(sampling_result['samples']['new_mismatch'], 'new template match failure')
    old_match_samples = format_samples(sampling_result['samples']['old_match'], 'old template match success')
    old_mismatch_samples = format_samples(sampling_result['samples']['old_mismatch'], 'old template match failure')

    prompt = TEMPLATE_SPLIT_PROMPT.format(
        old_template=old_template,
        new_template=new_template,
        new_match_rate=sampling_result['new_match_rate'],
        new_match_count=sampling_result['stats']['new_match_count'],
        old_match_rate=sampling_result['old_match_rate'],
        old_match_count=sampling_result['stats']['old_match_count'],
        total_count=sampling_result['total_count'],
        new_match_samples=new_match_samples,
        new_mismatch_samples=new_mismatch_samples,
        old_match_samples=old_match_samples,
        old_mismatch_samples=old_mismatch_samples
    )

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a log template analysis expert. Analyze log patterns and determine whether to refine template or split into multiple templates."
        )

        if repair_context:
            repair_context.add_stage_record(
                stage_name='2.2d-TemplateSplitAnalysis',
                llm_input=prompt,
                llm_output=response,
                conclusion='Pending verification'
            )

        result = {
            'decision': 'SPLIT',
            'analysis': '',
            'split_templates': [],
            'confidence': 'LOW',
            'raw_response': response
        }

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result['decision'] = parsed.get('decision', 'SPLIT').upper()
                result['analysis'] = parsed.get('analysis', '')
                result['split_templates'] = parsed.get('split_templates', [])
                result['confidence'] = parsed.get('confidence', 'LOW').upper()
            except json.JSONDecodeError:
                pass

        if result['split_templates']:
            verification_results = []
            for split_template in result['split_templates']:
                check_pattern = split_template.get('check_pattern', split_template.get('template', ''))
                if check_pattern:
                    check_pattern = check_pattern.rstrip()
                if check_pattern:
                    verify_result = check_pattern_by_event(
                        system_name, event_id, check_pattern, use_regex=False
                    )
                    verification_results.append({
                        'template': split_template.get('template', ''),
                        'check_pattern': check_pattern,
                        'match_count': verify_result.get('match_count', 0),
                        'match_rate': verify_result.get('match_rate', 0),
                        'total_count': verify_result.get('total_count', 0)
                    })
            result['verification_results'] = verification_results

            total_logs = sampling_result['total_count']
            total_covered = sum(v['match_count'] for v in verification_results)
            result['coverage_rate'] = total_covered / total_logs if total_logs > 0 else 0

        return result

    except Exception as e:
        return {
            'decision': 'SPLIT',
            'analysis': str(e),
            'split_templates': [],
            'confidence': 'LOW',
            'error': str(e),
            'raw_response': None
        }


def analyze_template_split_from_logs(llm_client, system_name, event_id, template,
                                      group_analysis, failed_samples, repair_context=None):
    gap_info = group_analysis.get('gap_info', {})
    length_analysis = f"""- Minimum parameter length: {gap_info.get('min_length', 'N/A')} characters
- Maximum parameter length: {gap_info.get('max_length', 'N/A')} characters
- Length difference: {gap_info.get('max_length', 0) - gap_info.get('min_length', 0)} characters
- Grouping threshold: {gap_info.get('threshold', 'N/A')} (max gap: {gap_info.get('gap_size', 'N/A')} characters)"""

    groups = group_analysis.get('groups', [])
    group_samples_text = ""
    for i, group in enumerate(groups, 1):
        group_samples_text += f"\n### Group {i} (parameter length {group.get('range', 'N/A')}, about {group.get('count', 0)} entries)\n"
        for sample in group.get('samples', [])[:5]:
            content = sample.get('Content', '')
            if len(content) > 150:
                content = content[:150] + '...'
            group_samples_text += f"- [LineId: {sample.get('LineId', 'N/A')}] {content}\n"

    failed_samples_text = ""
    for sample in failed_samples[:3]:
        gt = sample.get('ground_truth', '')
        gen = sample.get('generated_log', '')
        if len(gt) > 100:
            gt = gt[:100] + '...'
        if len(gen) > 100:
            gen = gen[:100] + '...'
        failed_samples_text += f"""- LineId: {sample.get('LineId', 'N/A')}
  Expected: {gt}
  Generated: {gen}
"""

    prompt = TEMPLATE_SPLIT_FROM_LOGS_PROMPT.format(
        template=template,
        length_analysis=length_analysis,
        group_samples=group_samples_text,
        failed_samples=failed_samples_text
    )

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a log template analysis expert. Analyze grouped logs and determine whether to split into multiple templates or use one template to cover all."
        )

        if repair_context:
            repair_context.add_stage_record(
                stage_name='2.2d-AnalyzeSplitFromLogs',
                llm_input=prompt,
                llm_output=response,
                conclusion='Pending verification'
            )

        result = {
            'decision': 'SPLIT',
            'analysis': '',
            'split_templates': [],
            'new_template': '',
            'variable_description': '',
            'confidence': 'LOW',
            'raw_response': response
        }

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result['decision'] = parsed.get('decision', 'SPLIT').upper()
                result['analysis'] = parsed.get('analysis', '')
                result['split_templates'] = parsed.get('split_templates', [])
                result['confidence'] = parsed.get('confidence', 'LOW').upper()
                result['new_template'] = parsed.get('new_template', '')
                result['variable_description'] = parsed.get('variable_description', '')
            except json.JSONDecodeError:
                pass

        if result['split_templates']:
            verification_results = []
            for split_template in result['split_templates']:
                check_pattern = split_template.get('check_pattern', split_template.get('template', ''))
                if check_pattern:
                    check_pattern = check_pattern.rstrip()
                if check_pattern:
                    verify_result = check_pattern_by_event(
                        system_name, event_id, check_pattern, use_regex=False
                    )
                    verification_results.append({
                        'template': split_template.get('template', ''),
                        'check_pattern': check_pattern,
                        'match_count': verify_result.get('match_count', 0),
                        'match_rate': verify_result.get('match_rate', 0),
                        'total_count': verify_result.get('total_count', 0)
                    })
            result['verification_results'] = verification_results

            total_logs = verification_results[0]['total_count'] if verification_results else 0
            total_covered = sum(v['match_count'] for v in verification_results)
            result['coverage_rate'] = min(total_covered / total_logs, 1.0) if total_logs > 0 else 0

        return result

    except Exception as e:
        return {
            'decision': 'GIVE_UP',
            'analysis': str(e),
            'split_templates': [],
            'confidence': 'LOW',
            'error': str(e),
            'raw_response': None
        }


def ask_llm_for_pattern_type(llm_client, template, group_analysis, repair_context=None):
    gap_info = group_analysis.get('gap_info', {})
    groups = group_analysis.get('groups', [])

    length_analysis = f"""- Minimum parameter length: {gap_info.get('min_length', 'N/A')} characters
- Maximum parameter length: {gap_info.get('max_length', 'N/A')} characters
- Number of groups: {len(groups)}"""

    group_samples_text = ""
    for i, group in enumerate(groups, 1):
        group_samples_text += f"\n### Group {i} (parameter length {group.get('range', 'N/A')}, about {group.get('count', 0)} entries)\n"
        for sample in group.get('samples', [])[:5]:
            content = sample.get('Content', '')
            group_samples_text += f"- [LineId: {sample.get('LineId', 'N/A')}] {content}\n"

    prompt = PATTERN_TYPE_JUDGMENT_PROMPT.format(
        template=template,
        length_analysis=length_analysis,
        group_samples=group_samples_text,
        PATTERN_TYPE_FEW_SHOTS=PATTERN_TYPE_FEW_SHOTS
    )

    try:
        response = llm_client.query(
            prompt=prompt,
            temperature=TEMPERATURE,
            system_prompt="You are a log template analysis expert. Determine if log parameters are variable length lists or need to be split into different templates."
        )

        if repair_context:
            repair_context.add_stage_record(
                stage_name='2.2d-Step3.5-PatternTypeJudgment',
                llm_input=prompt,
                llm_output=response,
                conclusion='Pending processing'
            )

        result = {
            'pattern_type': 'SPLIT',
            'analysis': '',
            'new_template': '',
            'variable_description': '',
            'split_reason': '',
            'confidence': 'LOW',
            'raw_response': response
        }

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result['pattern_type'] = parsed.get('pattern_type', 'SPLIT').upper()
                result['analysis'] = parsed.get('analysis', '')
                result['confidence'] = parsed.get('confidence', 'LOW').upper()

                if result['pattern_type'] == 'VARIABLE_LENGTH':
                    result['new_template'] = parsed.get('new_template', template)
                    result['variable_description'] = parsed.get('variable_description', '')
                else:
                    result['split_reason'] = parsed.get('split_reason', '')

            except json.JSONDecodeError:
                if 'VARIABLE_LENGTH' in response.upper():
                    result['pattern_type'] = 'VARIABLE_LENGTH'
                    result['analysis'] = response

        return result

    except Exception as e:
        return {
            'pattern_type': 'SPLIT',
            'analysis': str(e),
            'new_template': '',
            'variable_description': '',
            'split_reason': '',
            'confidence': 'LOW',
            'error': str(e),
            'raw_response': None
        }




class AutoRepairTool:
    def __init__(self, input_json_path, working_dataset_path=None, dry_run=True, output_dir=None, max_events=None, use_full_result=False, test_event=None, repair_template_path=None, target_system=None, model_name=None, api_key=None):
        self.input_json_path = input_json_path
        self.dry_run = dry_run
        self.max_events = max_events
        self.use_full_result = use_full_result
        self.test_event = test_event
        self.target_system = target_system


        self.model_name = model_name if model_name else MODEL_NAME
        self.api_key = api_key if api_key else get_api_key(self.model_name)
        self.model_tag = self.model_name.replace("-", "_").replace(".", "_")

        self.llm_client = LLMClient(model_type=self.model_name, api_key=self.api_key)

        self.few_shot_db = load_few_shot_examples()

        if use_full_result:
            self.all_samples, self.failed_samples, self.metadata = load_all_samples(input_json_path)
            print(f"[INFO] Using full test result mode: total samples {len(self.all_samples)}, failed {len(self.failed_samples)}")
        else:
            self.failed_samples, self.metadata = load_failed_samples(input_json_path)
            self.all_samples = None

        if target_system:
            print(f"[INFO] Filtering system: {target_system}")
            original_failed_count = len(self.failed_samples)
            self.failed_samples = [s for s in self.failed_samples if s.get('system_name') == target_system]
            if self.all_samples:
                original_all_count = len(self.all_samples)
                self.all_samples = [s for s in self.all_samples if s.get('system_name') == target_system]
                print(f"[INFO] Sample filtering: total {original_all_count} -> {len(self.all_samples)}, failed {original_failed_count} -> {len(self.failed_samples)}")
            else:
                print(f"[INFO] Failed sample filtering: {original_failed_count} -> {len(self.failed_samples)}")

            if not self.failed_samples:
                print(f"[Warning] System '{target_system}' has no failed samples!")

        if use_full_result and self.all_samples:
            self.grouped_samples = group_all_samples_by_event_id(self.all_samples)
        else:
            self.grouped_samples = None

        if target_system:
            self.system_name = target_system
        else:
            self.system_name = "all"

        if output_dir is None:
            output_dir = os.path.join(REPAIR_TEM_PATH, "output")

        if target_system:
            self.output_dir = os.path.join(output_dir, self.system_name)
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        if working_dataset_path:
            self.working_dataset_path = working_dataset_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if target_system:
                self.working_dataset_path = os.path.join(
                    self.output_dir,
                    f"working_dataset_{self.system_name}_{timestamp}.json"
                )
            else:
                self.working_dataset_path = os.path.join(
                    self.output_dir,
                    f"working_dataset_{self.model_tag}_{timestamp}.json"
                )

        self.run_log = {
            'start_time': datetime.now().isoformat(),
            'input_json': input_json_path,
            'use_full_result': use_full_result,
            'target_system': target_system,
            'model_name': self.model_name,
            'working_dataset': self.working_dataset_path,
            'output_dir': self.output_dir,
            'dry_run': dry_run,
            'max_events': max_events,
            'system_name': self.system_name,
            'total_samples': len(self.all_samples) if self.all_samples else None,
            'total_failed': len(self.failed_samples),
            'event_id_stats': [],
            'repairs': [],
            'commands_to_run': []
        }

        self._log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if target_system:
            self._log_path = os.path.join(self.output_dir, f"repair_run_log_{self.system_name}_{self.model_tag}_{self._log_timestamp}.json")
        else:
            self._log_path = os.path.join(self.output_dir, f"repair_run_log_{self.model_tag}_{self._log_timestamp}.json")

        self.repair_template_path = repair_template_path
        self.repair_template = None
        if repair_template_path and os.path.exists(repair_template_path):
            self.repair_template = load_repair_template(repair_template_path)
            print(f"[INFO] Loaded repair template: {repair_template_path}")
            print(f"       Total {len(self.repair_template.get('templates', []))} templates to process")

    def _get_sample_system_name(self, samples):
        if samples and len(samples) > 0:
            return samples[0].get('system_name', self.system_name)
        return self.system_name

    def prepare_working_dataset(self, source_dataset_path):
        if not os.path.exists(self.working_dataset_path):
            shutil.copy2(source_dataset_path, self.working_dataset_path)
            print(f"[INFO] Created working dataset: {self.working_dataset_path}")
        else:
            print(f"[INFO] Using existing working dataset: {self.working_dataset_path}")

    def _update_repair_template(self, event_id, updates, save=True):
        if not self.repair_template:
            return

        updated = update_repair_template_entry(self.repair_template, event_id, updates)

        if updated and save:
            update_repair_template_summary(self.repair_template)
            save_repair_template(self.repair_template_path, self.repair_template)
            print(f"  [repair_template] Updated repair record for {event_id}")

    def run(self):
        print("=" * 80)
        print("Auto Repair Failed Samples Tool")
        print("=" * 80)
        print(f"Input file: {self.input_json_path}")
        print(f"Model: {self.model_name}")
        if self.target_system:
            print(f"Target system: {self.target_system}")
        else:
            print(f"Target system: All systems (process separately)")
        print(f"System name: {self.system_name}")
        if self.use_full_result:
            print(f"Mode: Full test result (total samples {len(self.all_samples)}, failed {len(self.failed_samples)})")
        else:
            print(f"Mode: Failed samples only ({len(self.failed_samples)})")
        print(f"Output directory: {self.output_dir}")
        print(f"Dry Run mode: {self.dry_run}")
        if self.max_events:
            print(f"Max EventIds to process: {self.max_events}")
        if self.test_event:
            print(f"Test specific EventId: {self.test_event}")
        print(f"Real-time log file: {self._log_path}")
        print("=" * 80)

        if self.target_system:
            systems_to_process = [self.target_system]
        else:
            systems_to_process = sorted(set(s.get('system_name', 'Unknown') for s in self.failed_samples))
            print(f"\n[INFO] Detected {len(systems_to_process)} systems: {', '.join(systems_to_process)}")

        samples_by_system = {}
        for sample in self.failed_samples:
            sys_name = sample.get('system_name', 'Unknown')
            if sys_name not in samples_by_system:
                samples_by_system[sys_name] = []
            samples_by_system[sys_name].append(sample)

        all_samples_by_system = {}
        if self.use_full_result and self.all_samples:
            for sample in self.all_samples:
                sys_name = sample.get('system_name', 'Unknown')
                if sys_name not in all_samples_by_system:
                    all_samples_by_system[sys_name] = []
                all_samples_by_system[sys_name].append(sample)

        total_event_count = 0
        processed_event_count = 0

        for sys_idx, current_system in enumerate(systems_to_process, 1):
            print(f"\n{'#'*80}")
            print(f"# System [{sys_idx}/{len(systems_to_process)}]: {current_system}")
            print(f"{'#'*80}")

            system_failed_samples = samples_by_system.get(current_system, [])
            if not system_failed_samples:
                print(f"  [Skip] No failed samples for this system")
                continue

            print(f"\n[Step 1] Counting EventId frequency for {current_system}...")
            event_id_counts = count_event_ids(system_failed_samples)

            for eid, cnt in event_id_counts:
                self.run_log['event_id_stats'].append({
                    'system_name': current_system,
                    'EventId': eid,
                    'count': cnt
                })

            print(f"  Found {len(event_id_counts)} different EventIds with failed samples:")
            for event_id, count in event_id_counts[:10]:
                print(f"    {event_id}: {count} failures")
            if len(event_id_counts) > 10:
                print(f"    ... and {len(event_id_counts) - 10} more EventIds")

            total_event_count += len(event_id_counts)

            if self.max_events and self.max_events < len(event_id_counts):
                event_id_counts = event_id_counts[:self.max_events]
                print(f"\n  [INFO] Based on --max_events, only processing first {self.max_events} EventIds")

            if self.test_event:
                existing_event_ids = [eid for eid, _ in event_id_counts]
                if self.test_event in existing_event_ids:
                    target_count = next(cnt for eid, cnt in event_id_counts if eid == self.test_event)
                    event_id_counts = [(self.test_event, target_count)]
                    print(f"\n  [INFO] Based on --test_event, only processing specified EventId: {self.test_event}")
                    self.run_log['test_event'] = self.test_event
                else:
                    print(f"\n  [Skip] Specified EventId '{self.test_event}' not in this system")
                    continue

            grouped_failed = group_samples_by_event_id(system_failed_samples)

            if self.use_full_result:
                system_all_samples = all_samples_by_system.get(current_system, [])
                grouped_all = group_all_samples_by_event_id(system_all_samples)
            else:
                grouped_all = None

            print(f"\n[Step 2] Starting analysis and repair for {current_system} templates...")

            for idx, (event_id, count) in enumerate(event_id_counts, 1):
                print(f"\n{'='*80}")
                print(f"[{current_system}] [{idx}/{len(event_id_counts)}] Processing EventId: {event_id} ({count} failures)")
                print("=" * 80)

                if self.use_full_result and grouped_all and event_id in grouped_all:
                    group = grouped_all[event_id]
                    failed_samples = group['failed']
                    success_samples = group['success']
                    template = group['template']
                    print(f"  Success samples: {len(success_samples)}, Failed samples: {len(failed_samples)}")
                else:
                    failed_samples = grouped_failed[event_id]
                    success_samples = None
                    template = failed_samples[0]['template']

                repair_record = {
                    'system_name': current_system,
                    'event_id': event_id,
                    'template': template,
                    'failed_count': count,
                    'success_count': len(success_samples) if success_samples else 0,
                    'diagnosis': None,
                    'template_repair': None,
                    'pattern_check': None,
                    'test_results': [],
                    'description_repairs': [],
                    'regeneration_results': []
                }

                print("\n[2.1] Diagnosing error cause...")
                diagnosis = diagnose_error_cause(self.llm_client, event_id, template, failed_samples, success_samples)
                repair_record['diagnosis'] = diagnosis
                print(f"  Diagnosis result: {diagnosis['cause']}")
                print(f"  Confidence: {diagnosis.get('confidence', 'N/A')}")
                print(f"  Analysis: {diagnosis['analysis'][:200]}..." if len(diagnosis['analysis']) > 200 else f"  Analysis: {diagnosis['analysis']}")

                self._update_repair_template(event_id, {
                    'diagnosis': diagnosis,
                    'status': 'in_progress'
                })

                diagnosis_context = {
                    'diagnosis_input': diagnosis.get('diagnosis_input', ''),
                    'diagnosis_output': diagnosis.get('diagnosis_output', '')
                }

                print("\n[2.2] Entering repair state machine...")
                self._run_repair_state_machine(
                    event_id=event_id,
                    template=template,
                    failed_samples=failed_samples,
                    success_samples=success_samples,
                    repair_record=repair_record,
                    diagnosis=diagnosis,
                    diagnosis_context=diagnosis_context,
                    max_redirects=3
                )

                self.run_log['repairs'].append(repair_record)
                processed_event_count += 1

                # Update repair_template result
                # Determine status based on repair_record content
                final_status = 'completed'
                if repair_record.get('_give_up'):
                    final_status = 'failed'
                elif repair_record.get('diagnosis', {}).get('cause') == 'NONE':
                    final_status = 'skipped'

                self._update_repair_template(event_id, {
                    'template_repair': repair_record.get('template_repair'),
                    'pattern_check': repair_record.get('pattern_check'),
                    'test_results': repair_record.get('test_results', []),
                    'description_repairs': repair_record.get('description_repairs', []),
                    'regeneration_results': repair_record.get('regeneration_results', []),
                    'success_count': repair_record.get('success_count', 0),
                    'status': final_status
                })

                self._save_run_log_incremental()

            print(f"\n[INFO] {current_system} processing complete, processed {len(event_id_counts)} EventIds")

        self.run_log['total_event_count'] = total_event_count
        self.run_log['processed_event_count'] = processed_event_count
        self.run_log['systems_processed'] = systems_to_process

        self._save_run_log()

        print("\n" + "=" * 80)
        print("Repair process complete!")
        print(f"Processed {len(systems_to_process)} systems, {processed_event_count} EventIds")
        print("=" * 80)

    def _handle_template_error(self, event_id, template, samples, repair_record, diagnosis_context=None):
        print("\n[2.2a] Handling template error...")

        sample_system_name = self._get_sample_system_name(samples)

        print(f"  Getting all logs for {event_id}...")
        all_logs_data = get_all_logs_for_event(sample_system_name, event_id, output_dir=self.output_dir)

        if not all_logs_data:
            print("  [Warning] Unable to get log data, skipping template repair")
            return

        print(f"  Retrieved {all_logs_data.get('actual_count', 0)} logs")

        print("  Requesting LLM for template repair suggestion...")
        repair_suggestion = suggest_template_repair(
            self.llm_client, sample_system_name, event_id, template, samples, all_logs_data,
            diagnosis_context=diagnosis_context
        )
        repair_record['template_repair'] = repair_suggestion

        if repair_suggestion['needs_repair']:
            print(f"  Suggested repair:")
            print(f"    Old template: {repair_suggestion['old_template']}")
            print(f"    New template: {repair_suggestion['new_template']}")
            print(f"    Explanation: {repair_suggestion['explanation']}")
            print(f"    Confidence: {repair_suggestion.get('confidence', 'N/A')}")

            pattern_check_result = None
            if repair_suggestion.get('needs_check') and repair_suggestion.get('check_pattern'):
                check_pattern_cleaned = repair_suggestion['check_pattern'].rstrip()
                print(f"\n  [2.2a-1] Verifying pattern matches all logs...")
                print(f"    Check pattern: {check_pattern_cleaned}")
                print(f"    Is regex: {repair_suggestion.get('check_pattern_is_regex', False)}")

                pattern_check_result = check_pattern_by_event(
                    sample_system_name,
                    event_id,
                    check_pattern_cleaned,
                    use_regex=repair_suggestion.get('check_pattern_is_regex', False)
                )
                repair_record['pattern_check'] = pattern_check_result

                if pattern_check_result.get('error'):
                    print(f"    [Error] Verification failed: {pattern_check_result['error']}")
                elif pattern_check_result['all_match']:
                    print(f"    [Verification passed] All {pattern_check_result['total_count']} logs match the pattern")
                else:
                    print(f"    [Verification failed] {pattern_check_result['mismatch_count']}/{pattern_check_result['total_count']} logs don't match")
                    print(f"    Match rate: {pattern_check_result['match_rate']:.2%}")
                    if pattern_check_result.get('mismatch_samples'):
                        print("    Mismatched samples:")
                        for sample in pattern_check_result['mismatch_samples'][:3]:
                            print(f"      [LineId: {sample['LineId']}] {sample['Content']}")

                    repair_suggestion['needs_manual_review'] = True
                    print("\n    [Warning] Suggested template repair failed full verification, manual review needed")

            should_test = True
            if pattern_check_result and not pattern_check_result.get('all_match', True):
                should_test = False
                print("\n  [Skip test] Skipping repair effect test due to pattern verification failure")

            if should_test:
                print(f"\n  [2.2a-2] Testing repair effect ({len(samples)} failed samples)...")

                all_passed = True
                failed_tests = []

                for idx, test_sample in enumerate(samples, 1):
                    print(f"    Testing sample {idx}/{len(samples)} (LineId: {test_sample.get('LineId', 'unknown')})...", end=" ")

                    test_result = test_single_sample(
                        self.llm_client,
                        repair_suggestion['new_template'],
                        test_sample['description'],
                        test_sample['ground_truth'],
                        sample_system_name,
                        self.few_shot_db
                    )

                    repair_record['test_results'].append({
                        'type': 'template_repair',
                        'sample': test_sample,
                        'result': test_result
                    })

                    if test_result['match']:
                        print("Passed")
                    else:
                        print("Failed")
                        all_passed = False
                        failed_tests.append({
                            'line_id': test_sample.get('LineId', 'unknown'),
                            'expected': test_result['ground_truth'],
                            'generated': test_result['generated']
                        })

                # Only record command if all samples pass
                if all_passed:
                    print(f"\n  [Success] All {len(samples)} samples passed after repair!")

                    # Record repair command (not actually executed)
                    repair_cmd = (
                        f"python3 {REPAIR_TEM_PATH}/repair_template.py "
                        f"--system {sample_system_name} "
                        f"--event_id {event_id} "
                        f"--old_template \"{repair_suggestion['old_template']}\" "
                        f"--new_template \"{repair_suggestion['new_template']}\""
                    )
                    self.run_log['commands_to_run'].append({
                        'type': 'template_repair',
                        'event_id': event_id,
                        'command': repair_cmd,
                        'pattern_verified': pattern_check_result['all_match'] if pattern_check_result else None,
                        'confidence': repair_suggestion.get('confidence', 'LOW'),
                        'tested_samples_count': len(samples),
                        'all_tests_passed': True
                    })
                    print(f"\n  [Command recorded] {repair_cmd}")
                else:
                    print(f"\n  [Failed] {len(failed_tests)}/{len(samples)} samples failed after repair")
                    for ft in failed_tests[:3]:  # Show at most 3 failure details
                        print(f"    LineId {ft['line_id']}:")
                        print(f"      Expected: {ft['expected'][:80]}...")
                        print(f"      Generated: {ft['generated'][:80]}...")
        else:
            print("  LLM believes template does not need repair")

    def _handle_description_error(self, event_id, template, samples, repair_record, diagnosis_context=None):
        """Handle description error (Step 2.2b)

        Args:
            event_id: Event ID
            template: Template
            samples: Failed samples
            repair_record: Repair record
            diagnosis_context: Diagnosis context for context passing
        """
        print("\n[2.2b] Handling description error...")

        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(samples)

        for sample in samples[:3]:  # Process first 3 samples as examples
            line_id = sample.get('LineId', 'unknown')
            print(f"\n  Processing LineId: {line_id}")

            # Get context (output to specified directory)
            print(f"    Getting log context...")
            context_text = get_log_context(sample_system_name, event_id, line_id, context=5, output_dir=self.output_dir)

            if not context_text:
                print("    [Warning] Unable to get context, skipping")
                continue

            # Regenerate description (pass diagnosis context)
            print("    Requesting LLM to regenerate description...")
            new_desc = regenerate_description(
                self.llm_client,
                sample_system_name,
                event_id,
                line_id,
                template,
                sample['ground_truth'],
                sample['description'],
                context_text,
                diagnosis_context=diagnosis_context
            )

            print(f"    Old description: {sample['description'][:100]}...")
            print(f"    New description: {new_desc['new_description'][:100]}...")

            # Test new description
            print("    Testing new description...")
            test_result = test_single_sample(
                self.llm_client,
                template,
                new_desc['new_description'],
                sample['ground_truth'],
                sample_system_name,
                self.few_shot_db
            )

            desc_repair = {
                'line_id': line_id,
                'old_description': sample['description'],
                'new_description': new_desc['new_description'],
                'test_result': test_result
            }
            repair_record['description_repairs'].append(desc_repair)

            if test_result['match']:
                print("    [Success] New description test passed!")

                # Record modification (not actually executed)
                self.run_log['commands_to_run'].append({
                    'type': 'description_update',
                    'event_id': event_id,
                    'line_id': line_id,
                    'old_description': sample['description'],
                    'new_description': new_desc['new_description'],
                    'note': f"Update description in working dataset for EventId={event_id}, LineId={line_id}"
                })
            else:
                print("    [Failed] New description test failed")
                print(f"      Expected: {test_result['ground_truth']}")
                print(f"      Generated: {test_result['generated']}")

    def _handle_generator_error(self, event_id, template, failed_samples, success_samples, repair_record, diagnosis_context=None):
        """Handle generator error (Step 2.2c)

        Strategy: Use successful samples as few-shot examples to regenerate failed samples

        Args:
            event_id: Event ID
            template: Template
            failed_samples: Failed samples
            success_samples: Successful samples (used as few-shot)
            repair_record: Repair record
            diagnosis_context: Diagnosis context
        """
        print("\n[2.2c] Handling generator error (regeneration)...")

        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(failed_samples)

        # Build temporary few-shot: use successful samples from same template
        temp_few_shots = []
        if success_samples:
            for sample in success_samples[:3]:  # Take at most 3 successful samples as examples
                temp_few_shots.append({
                    'template': template,
                    'description': sample['description'],
                    'log': sample['ground_truth']
                })
            print(f"  Using {len(temp_few_shots)} successful samples as few-shot examples")

        regeneration_results = []

        for idx, sample in enumerate(failed_samples, 1):
            line_id = sample.get('LineId', 'unknown')
            print(f"\n  [{idx}/{len(failed_samples)}] Regenerating LineId: {line_id}")

            # Use successful samples as few-shot to regenerate
            test_result = test_single_sample(
                self.llm_client,
                template,
                sample['description'],
                sample['ground_truth'],
                sample_system_name,
                {sample_system_name: temp_few_shots} if temp_few_shots else self.few_shot_db
            )

            regen_record = {
                'line_id': line_id,
                'description': sample['description'],
                'ground_truth': sample['ground_truth'],
                'original_generated': sample.get('generated_log', ''),
                'new_generated': test_result.get('generated', ''),
                'success': test_result.get('match', False)
            }
            regeneration_results.append(regen_record)

            if test_result.get('match'):
                print(f"    [Success] Regeneration matched!")
            else:
                print(f"    [Failed] Regeneration still doesn't match")
                print(f"      Expected: {test_result['ground_truth'][:80]}...")
                print(f"      Generated: {test_result['generated'][:80] if test_result.get('generated') else 'None'}...")

        repair_record['regeneration_results'] = regeneration_results

        # Calculate success rate
        success_count = sum(1 for r in regeneration_results if r['success'])
        print(f"\n  Regeneration result: {success_count}/{len(regeneration_results)} succeeded")

        # Record to commands_to_run
        if success_count > 0:
            self.run_log['commands_to_run'].append({
                'type': 'generator_retry',
                'event_id': event_id,
                'total_samples': len(regeneration_results),
                'success_count': success_count,
                'success_rate': success_count / len(regeneration_results) if regeneration_results else 0,
                'note': f"Generator sporadic error, {success_count}/{len(regeneration_results)} succeeded after regeneration"
            })

    # =========================================================================
    # v2 version methods (support state machine transitions)
    # =========================================================================

    def _handle_template_error_v2(self, event_id, template, samples, repair_record,
                                   diagnosis_context, repair_context, remaining_redirects):
        """
        Handle template error (Step 2.2a) - Support state transitions

        Args:
            event_id: Event ID
            template: Template
            samples: Failed samples
            repair_record: Repair record
            diagnosis_context: Diagnosis context
            repair_context: RepairContext object for accumulating conversation history
            remaining_redirects: Remaining redirect count

        Returns:
            dict: {
                'status': RepairResult.*,
                'reason': str,
                'final_diagnosis': str,  # Only for GIVE_UP
                'suggestions': list      # Only for GIVE_UP
            }
        """
        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(samples)

        print(f"  ┌─ Getting all logs for {event_id}...")
        all_logs_data = get_all_logs_for_event(sample_system_name, event_id, output_dir=self.output_dir)

        if not all_logs_data:
            print("  └─ [Warning] Unable to get log data")
            return {
                'status': RepairResult.REDIRECT_DESCRIPTION,
                'reason': 'Unable to get log data, trying description repair'
            }

        print(f"  │  Retrieved {all_logs_data.get('actual_count', 0)} logs")
        print(f"  ├─ Requesting LLM for template repair suggestion...")
        repair_suggestion = suggest_template_repair(
            self.llm_client, sample_system_name, event_id, template, samples, all_logs_data,
            diagnosis_context=diagnosis_context,
            repair_context=repair_context
        )
        repair_record['template_repair'] = repair_suggestion

        # Check LLM confirmed cause
        confirmed_cause = repair_suggestion.get('confirmed_cause', 'TEMPLATE_ERROR')
        print(f"  │  LLM confirmed error type: {confirmed_cause}")

        # Case 1: LLM confirms it's a description problem
        if confirmed_cause == 'DESCRIPTION_ERROR':
            print(f"  └─ [Diagnosis correction] Confirmed problem is in description")
            # Update test conclusion in context
            if repair_context:
                repair_context.update_last_stage_test_results(
                    None, f'Diagnosis corrected to DESCRIPTION_ERROR: {repair_suggestion.get("explanation", "")}'
                )
            return {
                'status': RepairResult.REDIRECT_DESCRIPTION,
                'reason': repair_suggestion.get('explanation', 'Template is correct, problem is in description')
            }

        # Case 2: LLM confirms it's a generator problem
        if confirmed_cause == 'GENERATOR_ERROR':
            print(f"  └─ [Diagnosis correction] Confirmed it's a generator sporadic error")
            if repair_context:
                repair_context.update_last_stage_test_results(
                    None, f'Diagnosis corrected to GENERATOR_ERROR: {repair_suggestion.get("explanation", "")}'
                )
            return {
                'status': RepairResult.REDIRECT_GENERATOR,
                'reason': repair_suggestion.get('explanation', 'Template and description are correct, it is a generator sporadic error')
            }

        # Case 3: LLM confirms no repair needed
        if confirmed_cause == 'NONE' or not repair_suggestion.get('needs_repair'):
            print("  │  LLM believes template does not need repair")
            print("  ├─ Asking LLM to decide next step...")
            if repair_context:
                repair_context.update_last_stage_test_results(
                    None, f'Template does not need repair: {repair_suggestion.get("explanation", "")}'
                )
            # Ask LLM for next step
            redirect_decision = ask_llm_for_redirect_decision(
                self.llm_client, '2.2a-template-repair',
                f'Template does not need repair, but problem is still unresolved. Reason: {repair_suggestion.get("explanation", "")}',
                repair_context, remaining_redirects, samples
            )
            print(f"  └─ LLM decision: {redirect_decision['decision']}")
            return {
                'status': redirect_decision['decision'],
                'reason': redirect_decision['reason'],
                'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
                'suggestions': redirect_decision.get('suggestions', [])
            }

        # Case 4: needs_repair=true but template unchanged - ask LLM for next step (possibly VARIABLE_LENGTH scenario)
        if repair_suggestion['new_template'] == repair_suggestion['old_template']:
            print("  │  [Note] needs_repair=true but template unchanged")
            print("  │  Possible reasons: variable parameter count (VARIABLE_LENGTH), description issue, or other complex cases")
            print("  ├─ Asking LLM to decide next step...")
            if repair_context:
                repair_context.update_last_stage_test_results(
                    None, f'needs_repair=true but template unchanged: {repair_suggestion.get("explanation", "")}'
                )
            # Ask LLM for next step (including SPLIT option)
            redirect_decision = ask_llm_for_redirect_decision(
                self.llm_client, '2.2a-template-repair',
                f'LLM believes repair is needed (needs_repair=true) but could not provide a different new template. Analysis: {repair_suggestion.get("explanation", "")}',
                repair_context, remaining_redirects, samples
            )
            print(f"  └─ LLM decision: {redirect_decision['decision']}")
            return {
                'status': redirect_decision['decision'],
                'reason': redirect_decision['reason'],
                'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
                'suggestions': redirect_decision.get('suggestions', [])
            }

        # Case 5: Normal template repair flow
        print(f"  │")
        print(f"  ├─ [LLM Repair Suggestion]")
        print(f"  │  Old template: {repair_suggestion['old_template']}")
        print(f"  │  New template: {repair_suggestion['new_template']}")
        exp = repair_suggestion['explanation']
        print(f"  │  Explanation: {exp[:80]}{'...' if len(exp) > 80 else ''}")
        print(f"  │  Confidence: {repair_suggestion.get('confidence', 'N/A')}")

        # Step 2.2a-1: If verification needed, call check_all_logs
        pattern_check_result = None
        if repair_suggestion.get('needs_check') and repair_suggestion.get('check_pattern'):
            # Remove trailing spaces, LLM sometimes adds extra spaces at the end of pattern
            check_pattern_cleaned = repair_suggestion['check_pattern'].rstrip()
            print(f"  │")
            print(f"  ├─ [Pattern Verification] Checking if pattern matches all logs...")
            print(f"  │  Verification pattern: {check_pattern_cleaned}")

            pattern_check_result = check_pattern_by_event(
                sample_system_name,
                event_id,
                check_pattern_cleaned,
                use_regex=repair_suggestion.get('check_pattern_is_regex', False)
            )
            repair_record['pattern_check'] = pattern_check_result

            if pattern_check_result.get('error'):
                print(f"  │  [Error] Verification failed: {pattern_check_result['error']}")
            elif pattern_check_result['all_match']:
                print(f"  │  ✓ Verification passed: All {pattern_check_result['total_count']} logs match")
            else:
                # Pattern verification partial match, check if template split analysis is needed
                match_rate = pattern_check_result.get('match_rate', 0)
                print(f"  │  ✗ Verification failed: {pattern_check_result['mismatch_count']}/{pattern_check_result['total_count']} don't match")
                print(f"  │  New template match rate: {match_rate:.1%}")

                # Check if split analysis condition is met: 0 < new_match_rate < 1
                if 0 < match_rate < 1.0:
                    print(f"  │")
                    print(f"  ├─ [Split Detection] Partial match detected, performing sampling analysis...")

                    # Use new and old templates for sampling (remove trailing spaces)
                    new_check_pattern = repair_suggestion.get('check_pattern', repair_suggestion['new_template'])
                    if new_check_pattern:
                        new_check_pattern = new_check_pattern.rstrip()
                    old_check_pattern = template  # Use original template

                    sampling_result = check_dual_patterns_with_sampling(
                        sample_system_name, event_id,
                        new_check_pattern, old_check_pattern,
                        sample_count=2
                    )

                    if 'error' not in sampling_result:
                        old_match_rate = sampling_result.get('old_match_rate', 0)
                        print(f"  │  Old template match rate: {old_match_rate:.1%}")

                        # Check split condition: old_match_rate > 0
                        if old_match_rate > 0:
                            print(f"  └─ [Trigger Split] New template partial match and old template has matches")

                            # Save sampling result to repair record
                            repair_record['split_sampling'] = sampling_result

                            if repair_context:
                                repair_context.update_last_stage_test_results(
                                    pattern_check_result,
                                    f'Triggered template split analysis: new_rate={match_rate:.1%}, old_rate={old_match_rate:.1%}'
                                )

                            return {
                                'status': RepairResult.REDIRECT_SPLIT,
                                'reason': f'New template partial match ({match_rate:.1%}) and old template has matches ({old_match_rate:.1%}), split analysis needed',
                                'sampling_result': sampling_result,
                                'new_template': repair_suggestion['new_template'],
                                'old_template': template
                            }

                # Does not meet split condition, handle with original logic
                print(f"  ├─ Asking LLM to decide next step...")
                if repair_context:
                    repair_context.update_last_stage_test_results(
                        pattern_check_result,
                        f'Pattern verification failed: {pattern_check_result["mismatch_count"]}/{pattern_check_result["total_count"]} don\'t match'
                    )
                redirect_decision = ask_llm_for_redirect_decision(
                    self.llm_client, '2.2a-template-repair',
                    f'Pattern verification failed: {pattern_check_result["mismatch_count"]}/{pattern_check_result["total_count"]} logs don\'t match',
                    repair_context, remaining_redirects, samples
                )
                print(f"  └─ LLM decision: {redirect_decision['decision']}")
                return {
                    'status': redirect_decision['decision'],
                    'reason': redirect_decision['reason'],
                    'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
                    'suggestions': redirect_decision.get('suggestions', [])
                }

        # Step 2.2a-2: Test repair effect
        print(f"  │")
        print(f"  ├─ [Repair Effect Test] {len(samples)} failed samples")

        all_passed = True
        failed_tests = []
        test_results = []

        for idx, test_sample in enumerate(samples, 1):
            print(f"  │  Testing sample {idx}/{len(samples)} (LineId: {test_sample.get('LineId', 'unknown')})...", end=" ")

            test_result = test_single_sample(
                self.llm_client,
                repair_suggestion['new_template'],
                test_sample['description'],
                test_sample['ground_truth'],
                sample_system_name,
                self.few_shot_db
            )

            test_results.append({
                'type': 'template_repair',
                'sample': test_sample,
                'result': test_result
            })
            repair_record['test_results'].append(test_results[-1])

            if test_result['match']:
                print("✓ Passed")
            else:
                print("✗ Failed")
                all_passed = False
                failed_tests.append({
                    'line_id': test_sample.get('LineId', 'unknown'),
                    'expected': test_result['ground_truth'],
                    'generated': test_result['generated']
                })

        # Update test results in context
        if repair_context:
            repair_context.update_last_stage_test_results(
                test_results,
                f'Test result: {len(samples) - len(failed_tests)}/{len(samples)} passed'
            )

        # All samples passed
        if all_passed:
            print(f"  │")
            print(f"  └─ ✓ All {len(samples)} samples passed after repair!")

            # Record repair command
            repair_cmd = (
                f"python3 {REPAIR_TEM_PATH}/repair_template.py "
                f"--system {sample_system_name} "
                f"--event_id {event_id} "
                f"--old_template \"{repair_suggestion['old_template']}\" "
                f"--new_template \"{repair_suggestion['new_template']}\""
            )
            self.run_log['commands_to_run'].append({
                'type': 'template_repair',
                'event_id': event_id,
                'command': repair_cmd,
                'pattern_verified': pattern_check_result['all_match'] if pattern_check_result else None,
                'confidence': repair_suggestion.get('confidence', 'LOW'),
                'tested_samples_count': len(samples),
                'all_tests_passed': True
            })

            return {'status': RepairResult.CONTINUE, 'reason': 'Template repair successful, all tests passed'}

        # Test failed, ask LLM for next step
        print(f"  │")
        print(f"  ├─ ✗ {len(failed_tests)}/{len(samples)} samples failed after repair")
        print(f"  ├─ Asking LLM to decide next step...")
        redirect_decision = ask_llm_for_redirect_decision(
            self.llm_client, '2.2a-template-repair',
            f'Template repair test failed: {len(failed_tests)}/{len(samples)} samples failed',
            repair_context, remaining_redirects, samples
        )
        print(f"  └─ LLM decision: {redirect_decision['decision']}")

        return {
            'status': redirect_decision['decision'],
            'reason': redirect_decision['reason'],
            'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
            'suggestions': redirect_decision.get('suggestions', [])
        }

    def _handle_description_error_v2(self, event_id, template, samples, repair_record,
                                      diagnosis_context, repair_context, remaining_redirects):
        """
        Handle description error (Step 2.2b) - Support state transitions

        Args:
            event_id: Event ID
            template: Template
            samples: Failed samples
            repair_record: Repair record
            diagnosis_context: Diagnosis context
            repair_context: RepairContext object for accumulating conversation history
            remaining_redirects: Remaining redirect count

        Returns:
            dict: Result containing status field
        """
        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(samples)

        success_count = 0
        total_count = min(len(samples), 3)  # Process at most 3 samples

        for idx, sample in enumerate(samples[:3], 1):
            line_id = sample.get('LineId', 'unknown')
            is_last = (idx == total_count)
            prefix = "  └─" if is_last else "  ├─"
            cont_prefix = "     " if is_last else "  │  "

            print(f"{prefix} [Sample {idx}/{total_count}] LineId: {line_id}")

            # Get context
            print(f"{cont_prefix} Getting log context...")
            context_text = get_log_context(sample_system_name, event_id, line_id, context=5, output_dir=self.output_dir)

            if not context_text:
                print(f"{cont_prefix} ⚠ Unable to get context, skipping")
                continue

            # Regenerate description (pass accumulated context)
            print(f"{cont_prefix} Requesting LLM to regenerate description...")
            new_desc = regenerate_description(
                self.llm_client,
                sample_system_name,
                event_id,
                line_id,
                template,
                sample['ground_truth'],
                sample['description'],
                context_text,
                diagnosis_context=diagnosis_context,
                repair_context=repair_context
            )

            old_desc = sample['description'][:60]
            new_desc_text = new_desc['new_description'][:60]
            print(f"{cont_prefix} Old description: {old_desc}...")
            print(f"{cont_prefix} New description: {new_desc_text}...")

            # Test new description
            print(f"{cont_prefix} Testing new description...", end=" ")
            test_result = test_single_sample(
                self.llm_client,
                template,
                new_desc['new_description'],
                sample['ground_truth'],
                sample_system_name,
                self.few_shot_db
            )

            desc_repair = {
                'line_id': line_id,
                'old_description': sample['description'],
                'new_description': new_desc['new_description'],
                'test_result': test_result
            }
            repair_record['description_repairs'].append(desc_repair)

            if test_result['match']:
                print("✓ Passed")
                success_count += 1

                # Record modification
                self.run_log['commands_to_run'].append({
                    'type': 'description_update',
                    'event_id': event_id,
                    'line_id': line_id,
                    'old_description': sample['description'],
                    'new_description': new_desc['new_description'],
                    'note': f"Update description in working dataset for EventId={event_id}, LineId={line_id}"
                })
            else:
                print("✗ Failed")
                gt = test_result['ground_truth'][:60]
                gen = test_result['generated'][:60] if test_result.get('generated') else 'None'
                print(f"{cont_prefix}   Expected: {gt}...")
                print(f"{cont_prefix}   Generated: {gen}...")

        # Update test results in context
        if repair_context:
            repair_context.update_last_stage_test_results(
                repair_record['description_repairs'],
                f'Description repair result: {success_count}/{total_count} succeeded'
            )

        # Determine result
        print(f"\n  ╔{'═' * 50}")
        print(f"  ║ [Description Repair Summary]")
        print(f"  ╠{'═' * 50}")
        print(f"  ║ Samples processed: {total_count}")
        print(f"  ║ Success count: {success_count}")
        print(f"  ║ Success rate: {success_count}/{total_count} ({100*success_count//total_count if total_count > 0 else 0}%)")
        print(f"  ╚{'═' * 50}")

        if success_count == total_count and total_count > 0:
            print(f"\n  └─ ✓ All {total_count} description repair tests passed!")
            return {'status': RepairResult.CONTINUE, 'reason': f'Description repair successful, {success_count}/{total_count} tests passed'}

        if success_count > 0:
            # Partial success, also considered success
            print(f"\n  └─ ◐ Partial success: {success_count}/{total_count} description repair tests passed")
            return {'status': RepairResult.CONTINUE, 'reason': f'Description repair partially successful, {success_count}/{total_count} tests passed'}

        # All failed, ask LLM for next step
        print(f"\n  ├─ ✗ Description repair all failed (0/{total_count})")
        print(f"  └─ Asking LLM to decide next step...")
        redirect_decision = ask_llm_for_redirect_decision(
            self.llm_client, '2.2b-description-repair',
            f'Description repair all failed (0/{total_count})',
            repair_context, remaining_redirects, samples
        )

        return {
            'status': redirect_decision['decision'],
            'reason': redirect_decision['reason'],
            'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
            'suggestions': redirect_decision.get('suggestions', [])
        }

    def _handle_generator_error_v2(self, event_id, template, failed_samples, success_samples,
                                    repair_record, diagnosis_context, repair_context, remaining_redirects):
        """
        Handle generator error (Step 2.2c) - Support state transitions

        Args:
            event_id: Event ID
            template: Template
            failed_samples: Failed samples
            success_samples: Successful samples (used as few-shot)
            repair_record: Repair record
            diagnosis_context: Diagnosis context
            repair_context: RepairContext object
            remaining_redirects: Remaining redirect count

        Returns:
            dict: Result containing status field
        """
        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(failed_samples)

        # Build temporary few-shot: use successful samples from same template
        temp_few_shots = []
        if success_samples:
            for sample in success_samples[:3]:
                temp_few_shots.append({
                    'template': template,
                    'description': sample['description'],
                    'log': sample['ground_truth']
                })
            print(f"  ┌─ Building few-shot examples...")
            print(f"  │  Using {len(temp_few_shots)} successful samples as reference")
        else:
            print(f"  ┌─ No successful samples available, using default few-shot")

        regeneration_results = []
        total_samples = len(failed_samples)

        print(f"  │")
        print(f"  ├─ [Regeneration Test] {total_samples} failed samples")

        for idx, sample in enumerate(failed_samples, 1):
            line_id = sample.get('LineId', 'unknown')
            is_last = (idx == total_samples)
            prefix = "  │  └─" if is_last else "  │  ├─"
            cont_prefix = "  │     " if is_last else "  │  │  "

            print(f"{prefix} Sample {idx}/{total_samples} (LineId: {line_id})...", end=" ")

            # Use successful samples as few-shot to regenerate
            test_result = test_single_sample(
                self.llm_client,
                template,
                sample['description'],
                sample['ground_truth'],
                sample_system_name,
                {sample_system_name: temp_few_shots} if temp_few_shots else self.few_shot_db
            )

            regen_record = {
                'line_id': line_id,
                'description': sample['description'],
                'ground_truth': sample['ground_truth'],
                'original_generated': sample.get('generated_log', ''),
                'new_generated': test_result.get('generated', ''),
                'success': test_result.get('match', False)
            }
            regeneration_results.append(regen_record)

            if test_result.get('match'):
                print("✓ Passed")
            else:
                print("✗ Failed")
                gt = test_result['ground_truth'][:60]
                gen = test_result['generated'][:60] if test_result.get('generated') else 'None'
                print(f"{cont_prefix} Expected: {gt}...")
                print(f"{cont_prefix} Generated: {gen}...")

        repair_record['regeneration_results'] = regeneration_results

        # Calculate success rate
        success_count = sum(1 for r in regeneration_results if r['success'])
        total_regen = len(regeneration_results)

        print(f"  │")
        print(f"  ╔{'═' * 50}")
        print(f"  ║ [Generator Retry Summary]")
        print(f"  ╠{'═' * 50}")
        print(f"  ║ Retry samples: {total_regen}")
        print(f"  ║ Success count: {success_count}")
        print(f"  ║ Success rate: {success_count}/{total_regen} ({100*success_count//total_regen if total_regen > 0 else 0}%)")
        print(f"  ╚{'═' * 50}")

        # Record to context
        if repair_context:
            repair_context.add_stage_record(
                stage_name='2.2c-generator-retry',
                llm_input=f'Using {len(temp_few_shots)} successful samples as few-shot to regenerate {len(failed_samples)} failed samples',
                llm_output=f'Regeneration result: {success_count}/{total_regen} succeeded',
                conclusion=f'{success_count}/{total_regen} succeeded',
                test_results=regeneration_results
            )

        # Determine result
        if success_count == total_regen:
            # All succeeded
            self.run_log['commands_to_run'].append({
                'type': 'generator_retry',
                'event_id': event_id,
                'total_samples': total_regen,
                'success_count': success_count,
                'success_rate': 1.0,
                'note': f"Generator sporadic error, all succeeded after regeneration"
            })
            print(f"\n  └─ ✓ Regeneration all succeeded!")
            return {'status': RepairResult.CONTINUE, 'reason': 'Regeneration all succeeded'}

        if success_count > 0:
            # Partial success
            self.run_log['commands_to_run'].append({
                'type': 'generator_retry',
                'event_id': event_id,
                'total_samples': total_regen,
                'success_count': success_count,
                'success_rate': success_count / total_regen,
                'note': f"Generator sporadic error, {success_count}/{total_regen} succeeded after regeneration"
            })
            print(f"\n  └─ ◐ Partial success: {success_count}/{total_regen} regeneration passed")
            return {'status': RepairResult.CONTINUE, 'reason': f'Regeneration partially succeeded ({success_count}/{total_regen})'}

        # All failed, ask LLM for next step
        print(f"\n  ├─ ✗ Regeneration all failed (0/{total_regen})")
        print(f"  └─ Asking LLM to decide next step...")
        redirect_decision = ask_llm_for_redirect_decision(
            self.llm_client, '2.2c-generator-retry',
            f'Regeneration all failed (0/{total_regen})',
            repair_context, remaining_redirects, failed_samples
        )

        return {
            'status': redirect_decision['decision'],
            'reason': redirect_decision['reason'],
            'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
            'suggestions': redirect_decision.get('suggestions', [])
        }

    def _handle_template_split_v2(self, event_id, template, failed_samples, repair_record,
                                   diagnosis_context, repair_context, remaining_redirects,
                                   split_context):
        """
        Handle template split (Step 2.2d) - Only from 2.2a redirect

        When new template verification shows partial match, analyze if split into multiple templates is needed.

        Args:
            event_id: Event ID
            template: Original template
            failed_samples: Failed samples
            repair_record: Repair record
            diagnosis_context: Diagnosis context
            repair_context: RepairContext object
            remaining_redirects: Remaining redirect count
            split_context: Split context, containing sampling_result, new_template, old_template

        Returns:
            dict: Result containing status field
        """
        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(failed_samples)

        print("\n[2.2d] Template split analysis...")

        sampling_result = split_context.get('sampling_result', {})
        new_template = split_context.get('new_template', template)
        old_template = split_context.get('old_template', template)

        # Determine if new flow "analyze split from logs" is needed
        # Condition: no sampling data, or new_template == old_template (redirected from ask_llm_for_redirect_decision)
        use_new_flow = (not sampling_result or 'stats' not in sampling_result or
                        new_template == old_template)

        if use_new_flow:
            print(f"  ┌─ [New Flow] Analyze template split from log samples")
            print(f"  │")

            # Step 1: Uniform sampling of logs
            print(f"  ├─ [Step 1] Uniform sampling logs...")
            sample_result = self._get_uniform_sampled_logs(event_id, sample_count=50, system_name=sample_system_name)

            if 'error' in sample_result:
                print(f"  │  ✗ Sampling failed: {sample_result['error']}")
                # Don't give up directly, ask LLM for next step
                if remaining_redirects > 0:
                    print(f"  │")
                    print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                    stage_conclusion = f"Split analysis sampling failed: {sample_result['error']}"
                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, 'SPLIT', stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )
                    print(f"  └─ LLM decision: {redirect_decision['decision']}")
                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', 'SAMPLING_ERROR'),
                        'suggestions': redirect_decision.get('suggestions', ['Check if log data is accessible'])
                    }
                return {
                    'status': RepairResult.GIVE_UP,
                    'reason': f'Split analysis sampling failed: {sample_result["error"]}',
                    'final_diagnosis': 'SAMPLING_ERROR',
                    'suggestions': ['Check if log data is accessible']
                }

            sampled_logs = sample_result['logs']
            total_count = sample_result['total_count']
            print(f"  │  Total logs: {total_count}")
            print(f"  │  Sample count: {len(sampled_logs)}")
            print(f"  │  Sample step: {sample_result.get('sample_step', 'N/A')}")
            print(f"  │")

            # Step 2: Group by parameter length
            print(f"  ├─ [Step 2] Log length analysis...")
            group_analysis = self._group_logs_by_param_length(sampled_logs, template)

            gap_info = group_analysis.get('gap_info', {})
            groups = group_analysis.get('groups', [])

            # Calculate gap related values
            gap_size = gap_info.get('gap_size', 0)
            max_length = gap_info.get('max_length', 1)
            min_length = gap_info.get('min_length', 0)
            relative_gap = gap_size / max_length if max_length > 0 else 0

            # Record precheck info to repair_record (for debugging)
            split_precheck = {
                'timestamp': datetime.now().isoformat(),
                'sample_count': len(sampled_logs),
                'total_count': total_count,
                'min_length': min_length,
                'max_length': max_length,
                'length_diff': max_length - min_length,
                'gap_size': gap_size,
                'relative_gap': relative_gap,
                'threshold': gap_info.get('threshold', None),
                'group_count': len(groups),
                'groups': [
                    {
                        'range': g.get('range', ''),
                        'count': g.get('count', 0),
                        'avg_length': g.get('avg_length', 0),
                        'sample_logs': [s.get('Content', '')[:100] for s in g.get('samples', [])[:3]]
                    }
                    for g in groups
                ],
                'check1_pass': len(groups) > 1,
                'check2_pass': gap_size >= 10 or relative_gap >= 0.3,
                'all_param_lengths': group_analysis.get('all_lengths', [])[:100]  # Record first 100 length values
            }
            repair_record['split_precheck'] = split_precheck

            print(f"  │  Min param length: {min_length} chars")
            print(f"  │  Max param length: {max_length} chars")
            print(f"  │  Length difference: {max_length - min_length} chars")
            print(f"  │  Group threshold: {gap_info.get('threshold', 'N/A')} (max gap: {gap_size} chars)")
            print(f"  │  Relative gap: {relative_gap:.1%}")
            print(f"  │  Group count: {len(groups)}")

            for i, group in enumerate(groups, 1):
                print(f"  │    Group {i}: param length {group.get('range', 'N/A')}, ~{group.get('count', 0)} entries")

            # Step 3: Determine if LLM analysis is needed
            # If only one group or gap is small, split may not be needed
            if len(groups) <= 1:
                print(f"  │")
                print(f"  └─ ✗ Log length distribution is uniform, no split needed")
                print(f"  │  [Debug] Precheck 1 failed: group_count={len(groups)} <= 1")
                # Don't give up directly, ask LLM for next step
                if remaining_redirects > 0:
                    print(f"  │")
                    print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                    stage_conclusion = f"Split precheck failed: log length distribution is uniform (group_count={len(groups)})"
                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, 'SPLIT', stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )
                    print(f"  └─ LLM decision: {redirect_decision['decision']}")
                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', 'NO_SPLIT_NEEDED'),
                        'suggestions': redirect_decision.get('suggestions', ['Log structure is uniform, problem may be in description or generator'])
                    }
                return {
                    'status': RepairResult.GIVE_UP,
                    'reason': f'Log length distribution is uniform, cannot split (group_count={len(groups)})',
                    'final_diagnosis': 'NO_SPLIT_NEEDED',
                    'suggestions': ['Log structure is uniform, problem may be in description or generator'],
                    'precheck_failed': 'CHECK1_GROUP_COUNT'
                }

            # Conditions for determining if split is needed:
            # 1. gap_size >= 10 (absolute threshold, lowered to capture more split scenarios)
            # 2. or gap_size ratio to max length >= 30% (relative threshold)
            if gap_size < 10 and relative_gap < 0.3:
                print(f"  │")
                print(f"  └─ ✗ Log length gap is small (gap={gap_size}, relative ratio={relative_gap:.1%}), split not recommended")
                print(f"  │  [Debug] Precheck 2 failed: gap_size={gap_size} < 10 AND relative_gap={relative_gap:.1%} < 30%")
                # Don't give up directly, ask LLM for next step
                if remaining_redirects > 0:
                    print(f"  │")
                    print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                    stage_conclusion = f"Split precheck failed: log length gap is small (gap={gap_size}, relative ratio={relative_gap:.1%})"
                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, 'SPLIT', stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )
                    print(f"  └─ LLM decision: {redirect_decision['decision']}")
                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', 'SMALL_GAP'),
                        'suggestions': redirect_decision.get('suggestions', ['Log length difference is small, problem may be in description or generator'])
                    }
                return {
                    'status': RepairResult.GIVE_UP,
                    'reason': f'Log length gap is small (gap={gap_size}, relative ratio={relative_gap:.1%}), split not recommended',
                    'final_diagnosis': 'SMALL_GAP',
                    'suggestions': ['Log length difference is small, problem may be in description or generator'],
                    'precheck_failed': 'CHECK2_SMALL_GAP'
                }

            print(f"  │")
            # Step 4: Call LLM to analyze split (v1.12 integrates VARIABLE_LENGTH judgment)
            print(f"  ├─ [Step 3] Requesting LLM to analyze log patterns and provide solution...")
            split_analysis = analyze_template_split_from_logs(
                self.llm_client,
                sample_system_name,
                event_id,
                template,
                group_analysis,
                failed_samples,
                repair_context
            )

            # Append to split analysis list (don't overwrite previous records)
            split_analysis['source'] = 'analyze_template_split_from_logs'
            split_analysis['timestamp'] = datetime.now().isoformat()
            repair_record['split_analyses'].append(split_analysis)

            if split_analysis.get('error'):
                print(f"  │  ✗ Split analysis failed: {split_analysis['error']}")
                # Don't give up directly, ask LLM for next step
                if remaining_redirects > 0:
                    print(f"  │")
                    print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                    stage_conclusion = f"Split analysis failed: {split_analysis['error']}"
                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, 'SPLIT', stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )
                    print(f"  └─ LLM decision: {redirect_decision['decision']}")
                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', 'TEMPLATE_SPLIT_FAILED'),
                        'suggestions': redirect_decision.get('suggestions', ['Manual check of template split logic needed'])
                    }
                return {
                    'status': RepairResult.GIVE_UP,
                    'reason': f"Split analysis failed: {split_analysis['error']}",
                    'final_diagnosis': 'TEMPLATE_SPLIT_FAILED',
                    'suggestions': ['Manual check of template split logic needed']
                }

            decision = split_analysis.get('decision', 'SPLIT')
            analysis_text = split_analysis.get('analysis', '')[:150]
            print(f"  │  LLM judgment: {decision}")
            print(f"  │  Analysis: {analysis_text}{'...' if len(split_analysis.get('analysis', '')) > 150 else ''}")

            if decision == 'GIVE_UP':
                print(f"  │")
                print(f"  └─ ✗ LLM suggests giving up split")
                # Don't give up directly, ask LLM for next step
                if remaining_redirects > 0:
                    print(f"  │")
                    print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                    stage_conclusion = f"Split analysis LLM suggests giving up: {split_analysis.get('analysis', 'LLM suggests giving up')[:100]}"
                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, 'SPLIT', stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )
                    print(f"  └─ LLM decision: {redirect_decision['decision']}")
                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', 'LLM_GIVE_UP'),
                        'suggestions': redirect_decision.get('suggestions', ['LLM cannot determine split solution'])
                    }
                return {
                    'status': RepairResult.GIVE_UP,
                    'reason': split_analysis.get('analysis', 'LLM suggests giving up'),
                    'final_diagnosis': 'LLM_GIVE_UP',
                    'suggestions': ['LLM cannot determine split solution']
                }

            if decision == 'REFINE':
                # Refine template, no split needed
                print(f"  │")
                print(f"  ├─ ○ Conclusion: Template can be refined, no split needed")
                if split_analysis.get('split_templates'):
                    refined_template = split_analysis['split_templates'][0]
                    print(f"  │  Refined template: {refined_template.get('template', 'N/A')}")

                # Record as refinement suggestion
                self.run_log['commands_to_run'].append({
                    'type': 'template_refine',
                    'event_id': event_id,
                    'old_template': template,
                    'refined_template': split_analysis.get('split_templates', [{}])[0].get('template', ''),
                    'analysis': split_analysis.get('analysis', ''),
                    'note': 'Template refinement suggestion (not split)'
                })
                print(f"  └─ ✓ Template refinement complete")

                return {'status': RepairResult.CONTINUE, 'reason': 'Template can be refined, no split needed'}

            # v1.12: Add VARIABLE_LENGTH handling
            if decision == 'VARIABLE_LENGTH':
                # Variable length list pattern, generate new template with <*:list>
                new_template = split_analysis.get('new_template', template)
                variable_desc = split_analysis.get('variable_description', '')

                print(f"  │")
                print(f"  ╔{'═' * 50}")
                print(f"  ║ [Variable Length List Pattern]")
                print(f"  ╠{'═' * 50}")
                print(f"  ║ Original template: {template}")
                print(f"  ║ New template: {new_template}")
                print(f"  ║ Description: {variable_desc}")
                print(f"  ║ Confidence: {split_analysis.get('confidence', 'N/A')}")
                print(f"  ╚{'═' * 50}")

                # Record to commands_to_run
                self.run_log['commands_to_run'].append({
                    'type': 'template_variable_length',
                    'event_id': event_id,
                    'old_template': template,
                    'new_template': new_template,
                    'variable_description': variable_desc,
                    'analysis': split_analysis.get('analysis', ''),
                    'confidence': split_analysis.get('confidence', 'LOW'),
                    'note': 'Variable length list pattern, change <*> to <*:list>'
                })

                print(f"\n  └─ ✓ Identified as variable length list pattern!")

                return {
                    'status': RepairResult.CONTINUE,
                    'reason': f'Identified as variable length list pattern, new template: {new_template}'
                }

            # decision == 'SPLIT'
            print(f"  │")
            print(f"  ├─ ◎ Conclusion: Need to split into multiple templates")

            split_templates = split_analysis.get('split_templates', [])
            verification_results = split_analysis.get('verification_results', [])

            if not split_templates:
                print(f"  └─ ✗ LLM did not provide split templates")
                # Don't give up directly, ask LLM for next step
                if remaining_redirects > 0:
                    print(f"  │")
                    print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                    stage_conclusion = "LLM did not provide split templates"
                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, 'SPLIT', stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )
                    print(f"  └─ LLM decision: {redirect_decision['decision']}")
                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', 'TEMPLATE_SPLIT_NO_TEMPLATES'),
                        'suggestions': redirect_decision.get('suggestions', ['LLM determined split needed but did not provide specific templates'])
                    }
                return {
                    'status': RepairResult.GIVE_UP,
                    'reason': 'LLM did not provide split templates',
                    'final_diagnosis': 'TEMPLATE_SPLIT_NO_TEMPLATES',
                    'suggestions': ['LLM determined split needed but did not provide specific templates']
                }

            # Step 5: Display split results and verification
            print(f"  │")
            print(f"  ├─ [Step 4] Verifying split template coverage...")
            print(f"  │  Split templates ({len(split_templates)} total):")
            for i, tpl in enumerate(split_templates, 1):
                print(f"  │    Template {i}: {tpl.get('template', 'N/A')}")
                desc = tpl.get('description', 'N/A')[:50]
                print(f"  │      Description: {desc}{'...' if len(tpl.get('description', '')) > 50 else ''}")

            # Display verification results
            if verification_results:
                print(f"  │")
                print(f"  │  Verification results:")
                for i, vr in enumerate(verification_results, 1):
                    print(f"  │    Template {i}: matched {vr['match_count']}/{vr['total_count']} ({vr['match_rate']:.1%})")

                coverage_rate = split_analysis.get('coverage_rate', 0)
                print(f"  │  Combined coverage: {coverage_rate:.1%}")

                # Check if split is successful (coverage >= 90%)
                if coverage_rate >= 0.9:
                    print(f"  │  ✓ Coverage verification passed ({coverage_rate:.1%} >= 90%)")
                    print(f"  │")

                    # Step 5: Test failed samples for verification
                    print(f"  ├─ [Step 5] Testing failed samples ({len(failed_samples)} total)...")
                    test_results = self._test_samples_with_split_templates(
                        failed_samples, split_templates, verification_results
                    )
                    repair_record['split_test_results'] = test_results

                    success_count = sum(1 for r in test_results if r.get('match'))
                    print(f"  │  Test results: {success_count}/{len(test_results)} passed")

                    if success_count == 0:
                        print(f"  │")
                        print(f"  ├─ ✗ All tests failed after split")
                        print(f"  └─ Asking LLM to decide next step...")

                        # Build detailed test result comparison info
                        test_details = []
                        for r in test_results:
                            test_details.append(f"""- Sample {r.get('line_id')}:
  Template used: {r.get('template_used')}
  Expected output: {r.get('ground_truth', '')[:150]}...
  Actual generated: {r.get('generated', '')[:150]}...""")

                        stage_conclusion = f"""Template split coverage verification passed ({coverage_rate:.1%}), but all failed sample tests failed (0/{len(test_results)})

**Original template**: {template}
**Split templates**:
{chr(10).join([f"  - {t.get('template')}" for t in split_templates])}

**Test result details**:
{chr(10).join(test_details)}

**Analysis**: From the test results, the new template structure may be correct, but the generated content doesn't match expected.
This may be because:
1. The description is still based on the old template, need to update description to match new template
2. Or the split templates themselves have issues, need to reconsider

If choosing REDIRECT_DESCRIPTION, the system will use the new template and regenerate description."""

                        redirect_decision = ask_llm_for_redirect_decision(
                            self.llm_client, '2.2d-template-split',
                            stage_conclusion,
                            repair_context, remaining_redirects, failed_samples
                        )

                        # Determine suggested new template (choose the most used template in tests, excluding original)
                        template_usage = {}
                        for r in test_results:
                            tpl = r.get('template_used', '')
                            if tpl and tpl != template:
                                template_usage[tpl] = template_usage.get(tpl, 0) + 1
                        suggested_template = max(template_usage, key=template_usage.get) if template_usage else None

                        return {
                            'status': redirect_decision['decision'],
                            'reason': redirect_decision['reason'],
                            'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
                            'suggestions': redirect_decision.get('suggestions', []),
                            'suggested_template': suggested_template,  # Pass new template
                            'split_templates': split_templates  # Pass all split templates
                        }

                    print(f"  │")
                    print(f"  ╔{'═' * 50}")
                    print(f"  ║ [Split Successful]")
                    print(f"  ╠{'═' * 50}")
                    print(f"  ║ Coverage: {coverage_rate:.1%}")
                    print(f"  ║ Tests passed: {success_count}/{len(test_results)}")
                    print(f"  ╚{'═' * 50}")

                    # Record split results
                    split_record = {
                        'type': 'template_split',
                        'event_id': event_id,
                        'original_template': template,
                        'split_templates': [],
                        'coverage_rate': coverage_rate,
                        'test_success_rate': success_count / len(test_results) if test_results else 0,
                        'note': f'Split from log analysis successful, coverage {coverage_rate:.1%}, tests {success_count}/{len(test_results)} passed'
                    }

                    for i, (tpl, vr) in enumerate(zip(split_templates, verification_results)):
                        split_record['split_templates'].append({
                            'template': tpl.get('template', ''),
                            'check_pattern': tpl.get('check_pattern', ''),
                            'description': tpl.get('description', ''),
                            'match_count': vr['match_count']
                        })

                    self.run_log['commands_to_run'].append(split_record)
                    repair_record['split_result'] = split_record
                    print(f"\n  └─ ✓ Template split verification passed!")

                    return {'status': RepairResult.CONTINUE, 'reason': f'Template split successful, coverage {coverage_rate:.1%}, tests {success_count}/{len(test_results)} passed'}

                else:
                    print(f"  │  ✗ Coverage insufficient ({coverage_rate:.1%} < 90%)")

            # Split failed
            print(f"  │")
            print(f"  └─ ✗ Template split verification failed, coverage insufficient")
            # Don't give up directly, ask LLM for next step
            if remaining_redirects > 0:
                print(f"  │")
                print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                stage_conclusion = "Template split verification failed, coverage insufficient"
                redirect_decision = ask_llm_for_redirect_decision(
                    self.llm_client, 'SPLIT', stage_conclusion,
                    repair_context, remaining_redirects, failed_samples
                )
                print(f"  └─ LLM decision: {redirect_decision['decision']}")
                return {
                    'status': redirect_decision['decision'],
                    'reason': redirect_decision['reason'],
                    'final_diagnosis': redirect_decision.get('final_diagnosis', 'TEMPLATE_SPLIT_LOW_COVERAGE'),
                    'suggestions': redirect_decision.get('suggestions', ['Split templates cannot fully cover all logs, manual check needed'])
                }
            return {
                'status': RepairResult.GIVE_UP,
                'reason': f'Template split verification failed, coverage insufficient',
                'final_diagnosis': 'TEMPLATE_SPLIT_LOW_COVERAGE',
                'suggestions': ['Split templates cannot fully cover all logs, manual check needed']
            }

        # ========== Below is original flow (redirected from 2.2a pattern verification, has new_template != old_template) ==========
        print(f"  ┌─ [Original Flow] Redirected from 2.2a, comparing old and new templates")
        print(f"  │  Original template: {old_template}")
        print(f"  │  Suggested new template: {new_template}")

        # Print sampling statistics (if available)
        if 'stats' in sampling_result:
            print(f"  │")
            print(f"  ├─ Sampling statistics:")
            print(f"  │    New template matches: {sampling_result['stats'].get('new_match_count', 'N/A')}/{sampling_result.get('total_count', 'N/A')}")
            print(f"  │    Old template matches: {sampling_result['stats'].get('old_match_count', 'N/A')}/{sampling_result.get('total_count', 'N/A')}")

        # Priority attempt: directly use new template + old template as split combination
        # Logic: new template matches special cases, old template as fallback for the rest
        # Only attempt when stats has complete data (redirected from 2.2a)
        stats = sampling_result.get('stats', {})
        new_match_count = stats.get('new_match_count')
        old_match_count = stats.get('old_match_count')
        total_count = sampling_result.get('total_count', 0)

        # If new template and old template combination can cover all logs (old template as fallback)
        # Coverage = 100% (because old template matches all or most logs)
        if (new_match_count is not None and old_match_count is not None and
            old_match_count == total_count and new_match_count > 0 and new_template != old_template):
            print(f"  │")
            print(f"  ├─ [Priority Attempt] Directly use old+new template combination")
            print(f"  │    Template 1 (special): {new_template}")
            print(f"  │    Template 2 (fallback): {old_template}")

            # Verify combination coverage
            # New template matches go to new template, rest go to old template
            coverage_rate = 1.0  # Old template as fallback, covers all

            print(f"  │    Combined coverage: {coverage_rate:.1%}")

            if coverage_rate >= 0.9:
                print(f"  │    ✓ Coverage verification passed ({coverage_rate:.1%} >= 90%)")
                print(f"  │")

                # Build temporary split_templates for testing
                temp_split_templates = [
                    {'template': new_template, 'check_pattern': new_template, 'description': 'Special case template'},
                    {'template': old_template, 'check_pattern': old_template, 'description': 'General fallback template'}
                ]

                # Test failed samples for verification
                print(f"  ├─ [Test Verification] Testing failed samples ({len(failed_samples)} total)...")
                test_results = self._test_samples_with_split_templates(
                    failed_samples, temp_split_templates, []
                )
                repair_record['split_test_results'] = test_results

                success_count = sum(1 for r in test_results if r.get('match'))
                print(f"  │    Test results: {success_count}/{len(test_results)} passed")

                if success_count == 0:
                    print(f"  │    ✗ All tests failed, continuing with LLM analysis...")
                    # All tests failed, continue with LLM analysis flow
                    pass
                else:
                    print(f"  │")
                    print(f"  ╔{'═' * 50}")
                    print(f"  ║ [Direct Combination Successful]")
                    print(f"  ╠{'═' * 50}")
                    print(f"  ║ Coverage: {coverage_rate:.1%}")
                    print(f"  ║ Tests passed: {success_count}/{len(test_results)}")
                    print(f"  ╚{'═' * 50}")

                    # Record split results
                    split_record = {
                        'type': 'template_split',
                        'event_id': event_id,
                        'original_template': old_template,
                        'split_templates': [
                            {
                                'template': new_template,
                                'description': 'Special case template (logs matching new template)',
                                'match_count': new_match_count
                            },
                            {
                                'template': old_template,
                                'description': 'General fallback template (remaining logs)',
                                'match_count': total_count - new_match_count
                            }
                        ],
                        'coverage_rate': coverage_rate,
                        'test_success_rate': success_count / len(test_results) if test_results else 0,
                        'note': f'Using 2.2a old+new template combination, coverage {coverage_rate:.1%}, tests {success_count}/{len(test_results)} passed'
                    }

                    self.run_log['commands_to_run'].append(split_record)
                    repair_record['split_result'] = split_record
                    # Append to split analysis list
                    repair_record['split_analyses'].append({
                        'decision': 'SPLIT',
                        'analysis': 'Directly using new and old templates from 2.2a stage as split combination',
                        'method': 'direct_combination',
                        'source': 'direct_combination',
                        'timestamp': datetime.now().isoformat()
                    })

                    return {'status': RepairResult.CONTINUE, 'reason': f'Old+new template combination split successful, coverage {coverage_rate:.1%}, tests {success_count}/{len(test_results)} passed'}

        # If direct combination doesn't work, call LLM to analyze if split is needed
        print(f"  │")
        print(f"  ├─ [LLM Analysis] Requesting LLM to analyze template split...")
        split_analysis = analyze_template_split(
            self.llm_client,
            sample_system_name,
            event_id,
            old_template,
            new_template,
            sampling_result,
            repair_context
        )

        # Append to split analysis list (don't overwrite previous records)
        split_analysis['source'] = 'analyze_template_split'
        split_analysis['timestamp'] = datetime.now().isoformat()
        repair_record['split_analyses'].append(split_analysis)

        if split_analysis.get('error'):
            print(f"  │    ✗ Split analysis failed: {split_analysis['error']}")
            print(f"  └─ ✗ Give up")
            # Don't give up directly, ask LLM for next step
            if remaining_redirects > 0:
                print(f"  │")
                print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                stage_conclusion = f"Split analysis failed: {split_analysis['error']}"
                redirect_decision = ask_llm_for_redirect_decision(
                    self.llm_client, 'SPLIT', stage_conclusion,
                    repair_context, remaining_redirects, failed_samples
                )
                print(f"  └─ LLM decision: {redirect_decision['decision']}")
                return {
                    'status': redirect_decision['decision'],
                    'reason': redirect_decision['reason'],
                    'final_diagnosis': redirect_decision.get('final_diagnosis', 'TEMPLATE_SPLIT_FAILED'),
                    'suggestions': redirect_decision.get('suggestions', ['Manual check of template split logic needed'])
                }
            return {
                'status': RepairResult.GIVE_UP,
                'reason': f"Split analysis failed: {split_analysis['error']}",
                'final_diagnosis': 'TEMPLATE_SPLIT_FAILED',
                'suggestions': ['Manual check of template split logic needed']
            }

        decision = split_analysis.get('decision', 'SPLIT')
        analysis_text = split_analysis.get('analysis', '')[:150]
        print(f"  │    LLM judgment: {decision}")
        print(f"  │    Analysis: {analysis_text}{'...' if len(split_analysis.get('analysis', '')) > 150 else ''}")

        if decision == 'REFINE':
            # Refine template, no split needed
            print(f"  │")
            print(f"  ├─ ○ Conclusion: Template can be refined, no split needed")
            if split_analysis.get('split_templates'):
                refined_template = split_analysis['split_templates'][0]
                print(f"  │    Refined template: {refined_template.get('template', 'N/A')}")

            # Record as refinement suggestion
            self.run_log['commands_to_run'].append({
                'type': 'template_refine',
                'event_id': event_id,
                'old_template': old_template,
                'refined_template': split_analysis.get('split_templates', [{}])[0].get('template', ''),
                'analysis': split_analysis.get('analysis', ''),
                'note': 'Template refinement suggestion (not split)'
            })
            print(f"  └─ ✓ Template refinement complete")

            return {'status': RepairResult.CONTINUE, 'reason': 'Template can be refined, no split needed'}

        # v1.12: Add VARIABLE_LENGTH handling
        if decision == 'VARIABLE_LENGTH':
            # Variable length list pattern, generate new template with <*:list>
            new_template = split_analysis.get('new_template', old_template)
            variable_desc = split_analysis.get('variable_description', '')

            print(f"  │")
            print(f"  ╔{'═' * 50}")
            print(f"  ║ [Variable Length List Pattern]")
            print(f"  ╠{'═' * 50}")
            print(f"  ║ Original template: {old_template}")
            print(f"  ║ New template: {new_template}")
            print(f"  ║ Description: {variable_desc}")
            print(f"  ║ Confidence: {split_analysis.get('confidence', 'N/A')}")
            print(f"  ╚{'═' * 50}")

            # Record to commands_to_run
            self.run_log['commands_to_run'].append({
                'type': 'template_variable_length',
                'event_id': event_id,
                'old_template': old_template,
                'new_template': new_template,
                'variable_description': variable_desc,
                'analysis': split_analysis.get('analysis', ''),
                'confidence': split_analysis.get('confidence', 'LOW'),
                'note': 'Variable length list pattern, change <*> to <*:list>'
            })

            print(f"\n  └─ ✓ Identified as variable length list pattern!")

            return {
                'status': RepairResult.CONTINUE,
                'reason': f'Identified as variable length list pattern, new template: {new_template}'
            }

        # decision == 'SPLIT'
        print(f"  │")
        print(f"  ├─ ◎ Conclusion: Need to split into multiple templates")

        split_templates = split_analysis.get('split_templates', [])
        verification_results = split_analysis.get('verification_results', [])

        if not split_templates:
            print(f"  │    ✗ LLM did not provide split templates")
            print(f"  └─ ✗ Give up")
            # Don't give up directly, ask LLM for next step
            if remaining_redirects > 0:
                print(f"  │")
                print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
                stage_conclusion = "LLM did not provide split templates"
                redirect_decision = ask_llm_for_redirect_decision(
                    self.llm_client, 'SPLIT', stage_conclusion,
                    repair_context, remaining_redirects, failed_samples
                )
                print(f"  └─ LLM decision: {redirect_decision['decision']}")
                return {
                    'status': redirect_decision['decision'],
                    'reason': redirect_decision['reason'],
                    'final_diagnosis': redirect_decision.get('final_diagnosis', 'TEMPLATE_SPLIT_NO_TEMPLATES'),
                    'suggestions': redirect_decision.get('suggestions', ['LLM determined split needed but did not provide specific templates'])
                }
            return {
                'status': RepairResult.GIVE_UP,
                'reason': 'LLM did not provide split templates',
                'final_diagnosis': 'TEMPLATE_SPLIT_NO_TEMPLATES',
                'suggestions': ['LLM determined split needed but did not provide specific templates']
            }

        # Display split results
        print(f"  │")
        print(f"  │    Split templates ({len(split_templates)} total):")
        for i, tpl in enumerate(split_templates, 1):
            print(f"  │      Template {i}: {tpl.get('template', 'N/A')}")
            desc = tpl.get('description', 'N/A')[:50]
            print(f"  │        Description: {desc}{'...' if len(tpl.get('description', '')) > 50 else ''}")

        # Display verification results
        if verification_results:
            print(f"  │")
            print(f"  ├─ [Coverage Verification]")
            for i, vr in enumerate(verification_results, 1):
                print(f"  │    Template {i}: matched {vr['match_count']}/{vr['total_count']} ({vr['match_rate']:.1%})")

            coverage_rate = split_analysis.get('coverage_rate', 0)
            print(f"  │    Combined coverage: {coverage_rate:.1%}")

            # Check if split is successful (coverage >= 90%)
            if coverage_rate >= 0.9:
                print(f"  │    ✓ Coverage verification passed ({coverage_rate:.1%} >= 90%)")
                print(f"  │")

                # Test failed samples for verification
                print(f"  ├─ [Test Verification] Testing failed samples ({len(failed_samples)} total)...")
                test_results = self._test_samples_with_split_templates(
                    failed_samples, split_templates, verification_results
                )
                repair_record['split_test_results'] = test_results

                success_count = sum(1 for r in test_results if r.get('match'))
                print(f"  │    Test results: {success_count}/{len(test_results)} passed")

                if success_count == 0:
                    print(f"  │    ✗ All tests failed")
                    print(f"  │")
                    print(f"  └─ Asking LLM to decide next step...")

                    # Build detailed test result comparison info
                    test_details = []
                    for r in test_results:
                        test_details.append(f"""- Sample {r.get('line_id')}:
  Template used: {r.get('template_used')}
  Expected output: {r.get('ground_truth', '')[:150]}...
  Actual generated: {r.get('generated', '')[:150]}...""")

                    stage_conclusion = f"""Template split coverage verification passed ({coverage_rate:.1%}), but all failed sample tests failed (0/{len(test_results)})

**Original template**: {old_template}
**New template**: {new_template}
**Split templates**:
{chr(10).join([f"  - {t.get('template')}" for t in split_templates])}

**Test result details**:
{chr(10).join(test_details)}

**Analysis**: From the test results, the new template structure may be correct, but the generated content doesn't match expected.
This may be because:
1. The description is still based on the old template, need to update description to match new template
2. Or the split templates themselves have issues, need to reconsider

If choosing REDIRECT_DESCRIPTION, the system will use the new template and regenerate description."""

                    redirect_decision = ask_llm_for_redirect_decision(
                        self.llm_client, '2.2d-template-split',
                        stage_conclusion,
                        repair_context, remaining_redirects, failed_samples
                    )

                    # Determine suggested new template (choose the most used template in tests, excluding original)
                    template_usage = {}
                    for r in test_results:
                        tpl = r.get('template_used', '')
                        if tpl and tpl != old_template:
                            template_usage[tpl] = template_usage.get(tpl, 0) + 1
                    suggested_template = max(template_usage, key=template_usage.get) if template_usage else new_template

                    return {
                        'status': redirect_decision['decision'],
                        'reason': redirect_decision['reason'],
                        'final_diagnosis': redirect_decision.get('final_diagnosis', ''),
                        'suggestions': redirect_decision.get('suggestions', []),
                        'suggested_template': suggested_template,  # Pass new template
                        'split_templates': split_templates  # Pass all split templates
                    }

                print(f"  │")
                print(f"  ╔{'═' * 50}")
                print(f"  ║ [Split Successful]")
                print(f"  ╠{'═' * 50}")
                print(f"  ║ Coverage: {coverage_rate:.1%}")
                print(f"  ║ Tests passed: {success_count}/{len(test_results)}")
                print(f"  ╚{'═' * 50}")

                # Record split results
                split_record = {
                    'type': 'template_split',
                    'event_id': event_id,
                    'original_template': old_template,
                    'split_templates': [],
                    'coverage_rate': coverage_rate,
                    'test_success_rate': success_count / len(test_results) if test_results else 0,
                    'note': f'Template split successful, coverage {coverage_rate:.1%}, tests {success_count}/{len(test_results)} passed'
                }

                for i, (tpl, vr) in enumerate(zip(split_templates, verification_results)):
                    split_record['split_templates'].append({
                        'template': tpl.get('template', ''),
                        'check_pattern': tpl.get('check_pattern', ''),
                        'description': tpl.get('description', ''),
                        'match_count': vr['match_count'],
                        'samples': sampling_result['samples'].get('new_match' if i == 0 else 'new_mismatch', [])[:2]
                    })

                self.run_log['commands_to_run'].append(split_record)
                repair_record['split_result'] = split_record
                print(f"\n  └─ ✓ Template split verification passed!")

                return {'status': RepairResult.CONTINUE, 'reason': f'Template split successful, coverage {coverage_rate:.1%}, tests {success_count}/{len(test_results)} passed'}

            else:
                print(f"  │    ✗ Coverage insufficient ({coverage_rate:.1%} < 90%)")

        # Split failed
        print(f"  │")
        print(f"  └─ ✗ Template split verification failed, coverage insufficient")
        # Don't give up directly, ask LLM for next step
        if remaining_redirects > 0:
            print(f"  │")
            print(f"  ├─ [Ask Redirect] {remaining_redirects} redirect chances remaining, asking LLM for next step...")
            stage_conclusion = "Template split verification failed, coverage insufficient"
            redirect_decision = ask_llm_for_redirect_decision(
                self.llm_client, 'SPLIT', stage_conclusion,
                repair_context, remaining_redirects, failed_samples
            )
            print(f"  └─ LLM decision: {redirect_decision['decision']}")
            return {
                'status': redirect_decision['decision'],
                'reason': redirect_decision['reason'],
                'final_diagnosis': redirect_decision.get('final_diagnosis', 'TEMPLATE_SPLIT_LOW_COVERAGE'),
                'suggestions': redirect_decision.get('suggestions', ['Split templates cannot fully cover all logs, manual check needed'])
            }
        return {
            'status': RepairResult.GIVE_UP,
            'reason': f'Template split verification failed, coverage insufficient',
            'final_diagnosis': 'TEMPLATE_SPLIT_LOW_COVERAGE',
            'suggestions': ['Split templates cannot fully cover all logs, manual check needed']
        }

    def _run_repair_state_machine(self, event_id, template, failed_samples, success_samples,
                                   repair_record, diagnosis, diagnosis_context, max_redirects=3):
        """
        Run repair state machine

        Supports arbitrary transitions between 2.2a/b/c, with max redirect limit.

        Args:
            event_id: Event ID
            template: Template
            failed_samples: Failed samples
            success_samples: Successful samples (for GENERATOR_ERROR)
            repair_record: Repair record
            diagnosis: Diagnosis result
            diagnosis_context: Diagnosis context
            max_redirects: Max redirect count (default 3)
        """
        # Stage name mapping
        stage_names = {
            'TEMPLATE': '2.2a Template Repair',
            'DESCRIPTION': '2.2b Description Repair',
            'GENERATOR': '2.2c Generator Retry',
            'SPLIT': '2.2d Template Split'
        }

        def print_stage_banner(stage, is_initial=False):
            """Print stage entry banner"""
            name = stage_names.get(stage, stage)
            if is_initial:
                print(f"\n{'═' * 80}")
                print(f"║  [Initial Stage] {name}")
                print(f"║  Diagnosis: {diagnosis['cause']} | Confidence: {diagnosis.get('confidence', 'N/A')}")
                print(f"{'═' * 80}")
            else:
                print(f"\n{'─' * 80}")
                print(f"│  [Redirect Entry] {name}")
                print(f"{'─' * 80}")

        def print_redirect_banner(from_stage, to_stage, reason, redirect_num, remaining):
            """Print redirect banner"""
            from_name = stage_names.get(from_stage, from_stage)
            to_name = stage_names.get(to_stage, to_stage)
            print(f"\n{'▶' * 80}")
            print(f"  [State Redirect #{redirect_num}] {from_name} ──→ {to_name}")
            print(f"  Redirect reason: {reason[:100]}{'...' if len(reason) > 100 else ''}")
            print(f"  Remaining redirects: {remaining}")
            print(f"{'▶' * 80}")

        def print_result_banner(status, reason):
            """Print result banner"""
            if 'SUCCESS' in status:
                symbol = '✓'
                label = 'Repair Successful'
            elif status == 'GIVE_UP':
                symbol = '✗'
                label = 'Repair Abandoned'
            else:
                symbol = '!'
                label = status
            print(f"\n{'█' * 80}")
            print(f"  [{symbol} {label}]")
            print(f"  Status: {status}")
            print(f"  Reason: {reason[:150]}{'...' if len(reason) > 150 else ''}")
            print(f"{'█' * 80}")

        # Create accumulated context
        repair_context = RepairContext(
            event_id=event_id,
            template=template,
            failed_samples=failed_samples,
            success_samples=success_samples,
            diagnosis=diagnosis
        )

        # Determine initial state based on diagnosis
        initial_cause = diagnosis['cause']

        if initial_cause == DiagnosisResult.TEMPLATE_ERROR:
            current_stage = 'TEMPLATE'
        elif initial_cause == DiagnosisResult.DESCRIPTION_ERROR:
            current_stage = 'DESCRIPTION'
        elif initial_cause == DiagnosisResult.GENERATOR_ERROR:
            current_stage = 'GENERATOR'
        elif initial_cause == DiagnosisResult.BOTH:
            current_stage = 'TEMPLATE'  # BOTH starts from template
        else:
            print(f"\n{'─' * 80}")
            print(f"│  [Skip Repair] Diagnosis result is NONE")
            print(f"{'─' * 80}")
            repair_record['final_status'] = 'SKIPPED_NONE_DIAGNOSIS'
            return

        redirect_count = 0
        repair_record['redirect_history'] = []
        repair_record['redirect_decisions'] = []  # Record detailed info for each redirect decision
        repair_record['split_analyses'] = []  # Record all split analyses (may be multiple)
        repair_record['description_repairs'] = []  # Description repair records
        repair_record['template_repair'] = None  # Template repair record
        repair_record['pattern_check'] = None  # Pattern check result
        repair_record['test_results'] = []  # Test results

        # Print initial stage banner
        print_stage_banner(current_stage, is_initial=True)

        while redirect_count <= max_redirects:

            # Record redirect history
            repair_record['redirect_history'].append({
                'stage': current_stage,
                'redirect_count': redirect_count,
                'timestamp': datetime.now().isoformat()
            })

            # Calculate remaining redirects
            remaining_redirects = max_redirects - redirect_count

            # Get current template to use (may be updated by SPLIT stage)
            active_template = repair_record.get('_active_template', template)
            if active_template != template:
                print(f"  ├─ Using updated template: {active_template}")

            # Execute current stage
            if current_stage == 'TEMPLATE':
                result = self._handle_template_error_v2(
                    event_id, active_template, failed_samples, repair_record,
                    diagnosis_context, repair_context, remaining_redirects
                )

            elif current_stage == 'DESCRIPTION':
                result = self._handle_description_error_v2(
                    event_id, active_template, failed_samples, repair_record,
                    diagnosis_context, repair_context, remaining_redirects
                )

            elif current_stage == 'GENERATOR':
                result = self._handle_generator_error_v2(
                    event_id, active_template, failed_samples, success_samples,
                    repair_record, diagnosis_context, repair_context, remaining_redirects
                )

            elif current_stage == 'SPLIT':
                # Template split stage needs split_context
                split_context = repair_record.get('_split_context', {})
                result = self._handle_template_split_v2(
                    event_id, active_template, failed_samples, repair_record,
                    diagnosis_context, repair_context, remaining_redirects,
                    split_context
                )

            else:
                print(f"  [Error] Unknown stage: {current_stage}")
                repair_record['final_status'] = 'UNKNOWN_STAGE_ERROR'
                return

            # Process result
            if result['status'] == RepairResult.CONTINUE:
                final_status = f'SUCCESS_{current_stage}'
                final_reason = result.get('reason', '')
                print_result_banner(final_status, final_reason)
                repair_record['final_status'] = final_status
                repair_record['final_reason'] = final_reason
                # Save accumulated stage history
                repair_record['stage_history'] = repair_context.stage_history
                return

            elif result['status'] == RepairResult.GIVE_UP:
                print_result_banner('GIVE_UP', result.get('reason', ''))
                repair_record['final_status'] = 'GIVE_UP'
                repair_record['final_reason'] = result.get('reason', '')  # Fix: add final_reason
                repair_record['give_up_info'] = {
                    'reason': result.get('reason', ''),
                    'final_diagnosis': result.get('final_diagnosis', ''),
                    'suggestions': result.get('suggestions', []),
                    'precheck_failed': result.get('precheck_failed', '')  # Record which precheck failed
                }
                # Display suggestions
                if result.get('suggestions'):
                    print(f"  Suggestions:")
                    for s in result.get('suggestions', []):
                        print(f"    - {s}")
                # Save accumulated stage history
                repair_record['stage_history'] = repair_context.stage_history
                return

            elif result['status'] in [RepairResult.REDIRECT_TEMPLATE,
                                       RepairResult.REDIRECT_DESCRIPTION,
                                       RepairResult.REDIRECT_GENERATOR,
                                       RepairResult.REDIRECT_SPLIT]:
                # Redirect
                redirect_count += 1

                if redirect_count > max_redirects:
                    print_result_banner('MAX_REDIRECTS_REACHED', f'Reached max redirect limit ({max_redirects})')
                    repair_record['final_status'] = 'MAX_REDIRECTS_REACHED'
                    repair_record['final_reason'] = f'Reached max redirect limit ({max_redirects})'
                    # Save accumulated stage history
                    repair_record['stage_history'] = repair_context.stage_history
                    return

                # Determine next stage
                if result['status'] == RepairResult.REDIRECT_TEMPLATE:
                    next_stage = 'TEMPLATE'
                elif result['status'] == RepairResult.REDIRECT_DESCRIPTION:
                    next_stage = 'DESCRIPTION'
                elif result['status'] == RepairResult.REDIRECT_GENERATOR:
                    next_stage = 'GENERATOR'
                elif result['status'] == RepairResult.REDIRECT_SPLIT:
                    next_stage = 'SPLIT'
                    # Save split context for SPLIT stage
                    repair_record['_split_context'] = {
                        'sampling_result': result.get('sampling_result', {}),
                        'new_template': result.get('new_template', template),
                        'old_template': result.get('old_template', template)
                    }

                # If return contains suggested new template, save to _active_template
                # So subsequent stages (like DESCRIPTION) can use new template
                if result.get('suggested_template'):
                    repair_record['_active_template'] = result['suggested_template']

                # Print redirect banner
                print_redirect_banner(current_stage, next_stage, result.get('reason', ''),
                                     redirect_count, max_redirects - redirect_count)

                if result.get('suggested_template'):
                    print(f"  ├─ Updated active template to: {result['suggested_template']}")

                # Record detailed redirect decision info
                redirect_decision_record = {
                    'from_stage': current_stage,
                    'to_stage': next_stage,
                    'redirect_count': redirect_count,
                    'reason': result.get('reason', ''),
                    'analysis': result.get('analysis', ''),
                    'raw_response': result.get('raw_response', ''),
                    'timestamp': datetime.now().isoformat()
                }
                repair_record['redirect_decisions'].append(redirect_decision_record)

                # Print new stage entry banner
                print_stage_banner(next_stage)
                current_stage = next_stage

            else:
                print(f"  [Error] Unknown result status: {result['status']}")
                repair_record['final_status'] = 'UNKNOWN_RESULT_STATUS'
                # Save accumulated stage history
                repair_record['stage_history'] = repair_context.stage_history
                return

        # Exceeded loop (theoretically won't reach here)
        repair_record['final_status'] = 'UNEXPECTED_EXIT'
        # Save accumulated stage history
        repair_record['stage_history'] = repair_context.stage_history

    def _test_samples_with_split_templates(self, failed_samples, split_templates, verification_results):
        """
        Test failed samples with split templates

        For each failed sample:
        1. Determine which split template to use based on ground_truth
        2. Regenerate using that template
        3. Check if it matches

        Args:
            failed_samples: List of failed samples
            split_templates: List of split templates [{'template': ..., 'check_pattern': ..., 'description': ...}, ...]
            verification_results: Verification results for each template

        Returns:
            list: List of test results [{'line_id': ..., 'template_used': ..., 'match': bool, ...}, ...]
        """
        # Get actual system name from samples
        sample_system_name = self._get_sample_system_name(failed_samples)

        test_results = []

        # Build regex for each split template (for matching ground_truth)
        # Use wildcard_type='any' because log content may contain spaces or consecutive segments without spaces
        template_patterns = []
        for tpl in split_templates:
            template_str = tpl.get('template', '')
            check_pattern = tpl.get('check_pattern', template_str)
            # Remove trailing spaces
            if check_pattern:
                check_pattern = check_pattern.rstrip()
            # Convert check_pattern to regex (handle <*>)
            if '<*>' in check_pattern:
                regex = convert_template_to_regex(check_pattern, wildcard_type='any')
            else:
                regex = re.escape(check_pattern)
            template_patterns.append({
                'template': template_str,
                'regex': regex,
                'description': tpl.get('description', '')
            })

        for sample in failed_samples:
            line_id = sample.get('LineId', 'unknown')
            ground_truth = sample.get('ground_truth', '')
            description = sample.get('description', '')

            print(f"\n    [Testing LineId {line_id}]")

            # Determine which template to use (sorted by specificity: longer/more specific templates match first)
            # Sort by template length descending, let more specific templates match first
            sorted_patterns = sorted(template_patterns, key=lambda x: len(x['template']), reverse=True)

            matched_template = None
            matched_by_regex = None
            for pattern_info in sorted_patterns:
                try:
                    if re.match(pattern_info['regex'], ground_truth):
                        matched_template = pattern_info
                        matched_by_regex = pattern_info['regex']
                        break
                except re.error as e:
                    print(f"      ⚠ Regex match error: {e}")
                    continue

            if not matched_template:
                # No template matched, use first as fallback
                print(f"      ⚠ No template regex matched ground_truth, using first template as fallback")
                matched_template = template_patterns[0] if template_patterns else None

            if not matched_template:
                print("      ✗ No available template")
                test_results.append({
                    'line_id': line_id,
                    'template_used': None,
                    'match': False,
                    'error': 'No matching template'
                })
                continue

            # Print match info
            print(f"      Using template: {matched_template['template']}")
            print(f"      Template description: {matched_template['description']}")
            if matched_by_regex:
                print(f"      Match regex: {matched_by_regex[:80]}{'...' if len(matched_by_regex) > 80 else ''}")

            # Test using matched template
            test_result = test_single_sample(
                self.llm_client,
                matched_template['template'],
                description,
                ground_truth,
                sample_system_name,
                self.few_shot_db
            )

            match = test_result.get('match', False)
            generated = test_result.get('generated', '')

            # Print test result
            if match:
                print(f"      Result: ✅ Passed")
            else:
                print(f"      Result: ❌ Failed")
                # Print detailed comparison
                gt_display = ground_truth[:100] + '...' if len(ground_truth) > 100 else ground_truth
                gen_display = generated[:100] + '...' if len(generated) > 100 else generated
                print(f"      Expected: {gt_display}")
                print(f"      Generated: {gen_display}")
                # Simple difference description
                if len(ground_truth) != len(generated):
                    print(f"      Difference: Length mismatch (expected {len(ground_truth)} vs generated {len(generated)})")

            test_results.append({
                'line_id': line_id,
                'template_used': matched_template['template'],
                'template_description': matched_template['description'],
                'ground_truth': ground_truth,
                'generated': generated,
                'match': match
            })

        return test_results

    def _get_uniform_sampled_logs(self, event_id, sample_count=50, system_name=None):
        """
        Uniform sampling of logs (step = total // sample_count)

        Args:
            event_id: Event ID
            sample_count: Sample count (default 50)
            system_name: System name (v1.11 added, uses self.system_name if not specified)

        Returns:
            {
                'logs': [{'LineId': ..., 'Content': ...}, ...],
                'total_count': int,
                'sample_step': int,
                'error': str (if error)
            }
        """
        import csv

        # Use passed system_name or self.system_name
        actual_system_name = system_name if system_name else self.system_name

        # Build CSV file path
        csv_file_path = os.path.join(LOGHUB2_PATH, actual_system_name, f"{actual_system_name}_{event_id}_logs.csv")

        # If file doesn't exist, call get_all_logs_for_event to generate it
        if not os.path.exists(csv_file_path):
            print(f"  │  Log file doesn't exist, generating: {csv_file_path}")
            result = get_all_logs_for_event(actual_system_name, event_id)
            if result is None:
                return {'error': f"Unable to generate log file: {csv_file_path}", 'logs': [], 'total_count': 0}
            # Check again if file was generated successfully
            if not os.path.exists(csv_file_path):
                return {'error': f"Log file generation failed: {csv_file_path}", 'logs': [], 'total_count': 0}
            print(f"  │  Log file generated successfully")

        try:
            # First read all logs to get total count
            all_logs = []
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_logs.append({
                        'LineId': row.get('LineId', ''),
                        'Content': row.get('Content', '')
                    })

            total_count = len(all_logs)

            if total_count == 0:
                return {'error': 'Log file is empty', 'logs': [], 'total_count': 0}

            # Calculate sample step
            if total_count <= sample_count:
                # Total logs less than sample count, return all
                sampled_logs = all_logs
                sample_step = 1
            else:
                # Uniform sampling: step = total // sample_count
                sample_step = total_count // sample_count
                sampled_logs = []
                for i in range(0, total_count, sample_step):
                    if len(sampled_logs) < sample_count:
                        sampled_logs.append(all_logs[i])

            return {
                'logs': sampled_logs,
                'total_count': total_count,
                'sample_step': sample_step,
                'sample_count': len(sampled_logs)
            }

        except Exception as e:
            return {'error': f"Failed to read logs: {str(e)}", 'logs': [], 'total_count': 0}

    def _group_logs_by_param_length(self, logs, template):
        """
        Group by parameter length, using natural gap method to determine boundary

        Args:
            logs: Log list [{'LineId': ..., 'Content': ...}, ...]
            template: Original template (for extracting parameter part)

        Returns:
            {
                'groups': [
                    {
                        'range': '< 45' or '>= 45',
                        'count': int,
                        'samples': [typical sample list]
                    },
                    ...
                ],
                'gap_info': {
                    'threshold': 45,
                    'gap_size': 80,
                    'min_length': 4,
                    'max_length': 92
                },
                'length_details': [(log, length), ...]  # Each log and its parameter length
            }
        """
        # 1. Calculate template fixed part length (excluding <*>)
        # Parameter length = log total length - template fixed part length
        template_fixed_len = len(template.replace('<*>', ''))
        length_details = []  # [(log, param_length), ...]

        for log in logs:
            content = log.get('Content', '')
            # Parameter length = log total length - template fixed part length
            param_length = len(content) - template_fixed_len
            if param_length < 0:
                param_length = 0  # Prevent negative
            length_details.append((log, param_length))

        if not length_details:
            return {
                'groups': [],
                'gap_info': {'threshold': 0, 'gap_size': 0, 'min_length': 0, 'max_length': 0},
                'length_details': [],
                'all_lengths': []
            }

        # 2. Sort and deduplicate, find max gap
        all_lengths = [l for _, l in length_details]
        sorted_unique_lengths = sorted(set(all_lengths))

        min_length = min(all_lengths)
        max_length = max(all_lengths)

        # If only one length value, no grouping needed
        if len(sorted_unique_lengths) <= 1:
            return {
                'groups': [{
                    'range': f'= {sorted_unique_lengths[0]}' if sorted_unique_lengths else 'None',
                    'count': len(length_details),
                    'samples': [log for log, _ in length_details[:5]]
                }],
                'gap_info': {'threshold': 0, 'gap_size': 0, 'min_length': min_length, 'max_length': max_length},
                'length_details': length_details,
                'all_lengths': sorted(all_lengths)
            }

        # Calculate adjacent length differences, find max gap
        gaps = []
        for i in range(1, len(sorted_unique_lengths)):
            gap = sorted_unique_lengths[i] - sorted_unique_lengths[i-1]
            gaps.append((gap, sorted_unique_lengths[i-1], sorted_unique_lengths[i]))

        # Find max gap
        max_gap = max(gaps, key=lambda x: x[0])
        gap_size = max_gap[0]
        threshold = (max_gap[1] + max_gap[2]) / 2  # Boundary = gap midpoint

        # 3. Group by gap
        group1_logs = [(log, l) for log, l in length_details if l < threshold]
        group2_logs = [(log, l) for log, l in length_details if l >= threshold]

        # 4. Get typical samples for each group (first, middle, last after sorting by length)
        def get_typical_samples(group_logs, max_samples=5):
            if not group_logs:
                return []
            sorted_by_len = sorted(group_logs, key=lambda x: x[1])
            n = len(sorted_by_len)
            if n <= max_samples:
                return [log for log, _ in sorted_by_len]
            indices = [0, n//4, n//2, 3*n//4, n-1]
            return [sorted_by_len[i][0] for i in indices]

        groups = []
        if group1_logs:
            groups.append({
                'range': f'< {int(threshold)}',
                'count': len(group1_logs),
                'avg_length': sum(l for _, l in group1_logs) / len(group1_logs),
                'samples': get_typical_samples(group1_logs)
            })
        if group2_logs:
            groups.append({
                'range': f'>= {int(threshold)}',
                'count': len(group2_logs),
                'avg_length': sum(l for _, l in group2_logs) / len(group2_logs),
                'samples': get_typical_samples(group2_logs)
            })

        return {
            'groups': groups,
            'gap_info': {
                'threshold': threshold,
                'gap_size': gap_size,
                'min_length': min_length,
                'max_length': max_length
            },
            'length_details': length_details,
            'all_lengths': sorted(all_lengths)  
        }

    def _reorganize_repair_record(self, repair_record):
        stage_history = repair_record.get('stage_history', [])
        redirect_history = repair_record.get('redirect_history', [])
        redirect_decisions = repair_record.get('redirect_decisions', [])
        description_repairs = repair_record.get('description_repairs', [])
        split_analyses = repair_record.get('split_analyses', [])

        timeline = []

        for i, rh in enumerate(redirect_history):
            entry = {
                'step': i + 1,
                'timestamp': rh.get('timestamp'),
                'stage': rh.get('stage'),
                'redirect_count': rh.get('redirect_count', 0)
            }

            if i < len(stage_history):
                sh = stage_history[i]
                entry['stage_name'] = sh.get('stage', '')
                entry['conclusion'] = sh.get('conclusion', '')
                if sh.get('output'):
                    output = sh['output']
                    if '```json' in output:
                        try:
                            json_start = output.index('```json') + 7
                            json_end = output.index('```', json_start)
                            json_str = output[json_start:json_end].strip()
                            entry['llm_analysis'] = json.loads(json_str)
                        except:
                            entry['llm_output_preview'] = output[:300] + '...' if len(output) > 300 else output
                    else:
                        entry['llm_output_preview'] = output[:300] + '...' if len(output) > 300 else output

            for rd in redirect_decisions:
                if rd.get('redirect_count') == rh.get('redirect_count') + 1:
                    entry['next_action'] = rd.get('to_stage')
                    entry['redirect_reason'] = rd.get('reason', '')[:200]
                    break

            timeline.append(entry)

        summary = {
            'total_steps': len(timeline),
            'total_redirects': len(redirect_decisions),
            'stages_visited': [t.get('stage') for t in timeline],
            'description_repairs_count': len(description_repairs),
            'split_analyses_count': len(split_analyses),
            'final_status': repair_record.get('final_status'),
            'final_reason': repair_record.get('final_reason')
        }

        cleaned_record = {
            'event_id': repair_record.get('event_id'),
            'template': repair_record.get('template'),
            'failed_count': repair_record.get('failed_count'),
            'success_count': repair_record.get('success_count'),

            'diagnosis': {
                'cause': repair_record.get('diagnosis', {}).get('cause'),
                'confidence': repair_record.get('diagnosis', {}).get('confidence'),
                'analysis': repair_record.get('diagnosis', {}).get('analysis', '')[:500]
            } if repair_record.get('diagnosis') else None,

            'summary': summary,

            'repair_timeline': timeline,

            'details': {
                'description_repairs': description_repairs,
                'split_analyses': [
                    {
                        'decision': sa.get('decision'),
                        'analysis': sa.get('analysis', '')[:200],
                        'templates_count': len(sa.get('split_templates', [])),
                        'coverage_rate': sa.get('coverage_rate'),
                        'timestamp': sa.get('timestamp')
                    }
                    for sa in split_analyses
                ],
                'split_test_results': repair_record.get('split_test_results', []),
                'template_repair': repair_record.get('template_repair')
            }
        }

        return cleaned_record

    def _save_run_log_incremental(self):
        self.run_log['last_update_time'] = datetime.now().isoformat()

        output_log = self.run_log.copy()
        output_log['repairs'] = [
            self._reorganize_repair_record(r) for r in self.run_log.get('repairs', [])
        ]

        with open(self._log_path, 'w', encoding='utf-8') as f:
            json.dump(output_log, f, ensure_ascii=False, indent=2)

    def _save_run_log(self):
        self.run_log['end_time'] = datetime.now().isoformat()

        output_log = self.run_log.copy()
        output_log['repairs'] = [
            self._reorganize_repair_record(r) for r in self.run_log.get('repairs', [])
        ]

        with open(self._log_path, 'w', encoding='utf-8') as f:
            json.dump(output_log, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Run log saved to: {self._log_path}")

        txt_log_path = self._log_path.replace('.json', '.txt')
        self._save_readable_txt_log(txt_log_path, output_log)
        print(f"[INFO] Text log saved to: {txt_log_path}")

        self._save_repair_summary_json(self._log_timestamp)
        self._save_repair_summary_txt(self._log_timestamp)

    def _save_readable_txt_log(self, txt_path, output_log):
        """Save readable text log"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Auto Repair Run Log\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Start time: {output_log['start_time']}\n")
            f.write(f"End time: {output_log.get('end_time', 'N/A')}\n")
            f.write(f"System name: {output_log['system_name']}\n")
            f.write(f"Total failed samples: {output_log['total_failed']}\n")
            f.write(f"Output directory: {output_log['output_dir']}\n")
            f.write(f"Dry Run: {output_log['dry_run']}\n")
            if output_log.get('max_events'):
                f.write(f"Max EventIds to process: {output_log['max_events']}\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("EventId Statistics\n")
            f.write("-" * 80 + "\n")
            for stat in output_log['event_id_stats']:
                f.write(f"  {stat['EventId']}: {stat['count']} times\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Repair Process Details\n")
            f.write("=" * 80 + "\n")

            for repair in output_log.get('repairs', []):
                f.write(f"\n{'─' * 80}\n")
                f.write(f"EventId: {repair.get('event_id')}\n")
                f.write(f"Template: {repair.get('template')}\n")
                f.write(f"Failed sample count: {repair.get('failed_count')}\n")

                diagnosis = repair.get('diagnosis', {})
                if diagnosis:
                    f.write(f"\n[Diagnosis] Cause: {diagnosis.get('cause')} | Confidence: {diagnosis.get('confidence')}\n")

                summary = repair.get('summary', {})
                f.write(f"\n[Summary]\n")
                f.write(f"  Total steps: {summary.get('total_steps', 0)}\n")
                f.write(f"  Redirect count: {summary.get('total_redirects', 0)}\n")
                f.write(f"  Final status: {summary.get('final_status')}\n")
                if summary.get('final_reason'):
                    f.write(f"  Reason: {summary.get('final_reason')}\n")

                timeline = repair.get('repair_timeline', [])
                if timeline:
                    f.write(f"\n[Repair Timeline]\n")
                    for step in timeline:
                        stage_name = step.get('stage_name', step.get('stage', ''))
                        f.write(f"\n  ┌─ Step {step.get('step')}: {stage_name}\n")
                        f.write(f"  │  Time: {step.get('timestamp')}\n")
                        if step.get('conclusion'):
                            f.write(f"  │  Conclusion: {step.get('conclusion')}\n")
                        if step.get('next_action'):
                            f.write(f"  │  Next step: → {step.get('next_action')}\n")
                        if step.get('redirect_reason'):
                            reason = step.get('redirect_reason')[:100]
                            f.write(f"  │  Redirect reason: {reason}...\n")
                        f.write(f"  └─\n")

                f.write(f"{'─' * 80}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Commands to Execute\n")
            f.write("=" * 80 + "\n")
            commands = output_log.get('commands_to_run', [])
            if commands:
                for cmd_info in commands:
                    f.write(f"\n[{cmd_info['type']}] EventId: {cmd_info['event_id']}\n")
                    if 'command' in cmd_info:
                        f.write(f"  Command: {cmd_info['command']}\n")
                    if 'note' in cmd_info:
                        f.write(f"  Note: {cmd_info['note']}\n")
            else:
                f.write("\n  (No commands to execute)\n")

            f.write("\n" + "=" * 80 + "\n")

    def _save_repair_summary_json(self, timestamp):
        """Save repair summary file (JSON format, for programmatic reading)"""
        summary_path = os.path.join(self.output_dir, f"repair_summary_{self.system_name}_{timestamp}.json")

        failed_samples_by_event = {}
        for sample in self.failed_samples:
            event_id = sample.get('EventId', '')
            if event_id not in failed_samples_by_event:
                failed_samples_by_event[event_id] = []
            failed_samples_by_event[event_id].append(sample)

        template_repairs = []
        description_repairs = []

        quick_template_repairs = []
        quick_description_repairs = {}  # Aggregate by event_id

        for repair in self.run_log.get('repairs', []):
            event_id = repair.get('event_id', '')

            template_repair = repair.get('template_repair', {})
            if template_repair and template_repair.get('needs_repair'):
                test_passed = False
                for test in repair.get('test_results', []):
                    if test.get('type') == 'template_repair' and test.get('result', {}).get('match'):
                        test_passed = True
                        break

                failed_logs = []
                if event_id in failed_samples_by_event:
                    for sample in failed_samples_by_event[event_id]:
                        failed_logs.append(sample.get('ground_truth', ''))

                template_repairs.append({
                    'event_id': event_id,
                    'old_template': template_repair.get('old_template', ''),
                    'new_template': template_repair.get('new_template', ''),
                    'explanation': template_repair.get('explanation', ''),
                    'confidence': template_repair.get('confidence', ''),
                    'pattern_verified': (repair.get('pattern_check') or {}).get('all_match', None),
                    'test_passed': test_passed
                })

                quick_template_repairs.append({
                    'event_id': event_id,
                    'test_passed': test_passed,
                    'old_template': template_repair.get('old_template', ''),
                    'new_template': template_repair.get('new_template', ''),
                    'failed_logs': failed_logs
                })

            for desc_repair in repair.get('description_repairs', []):
                test_result = desc_repair.get('test_result', {})
                line_id = desc_repair.get('line_id', '')

                log_text = ''
                if event_id in failed_samples_by_event:
                    for sample in failed_samples_by_event[event_id]:
                        if str(sample.get('LineId', '')) == str(line_id):
                            log_text = sample.get('ground_truth', '')
                            break

                description_repairs.append({
                    'event_id': event_id,
                    'line_id': line_id,
                    'old_description': desc_repair.get('old_description', ''),
                    'new_description': desc_repair.get('new_description', ''),
                    'test_passed': test_result.get('match', False)
                })

                if event_id not in quick_description_repairs:
                    quick_description_repairs[event_id] = {
                        'event_id': event_id,
                        'failed_samples': []
                    }
                quick_description_repairs[event_id]['failed_samples'].append({
                    'line_id': line_id,
                    'log_text': log_text,
                    'old_description': desc_repair.get('old_description', ''),
                    'new_description': desc_repair.get('new_description', ''),
                    'test_passed': test_result.get('match', False)
                })

        quick_summary = {
            'template_repairs_count': len(quick_template_repairs),
            'description_repairs_count': len(description_repairs),
            'template_repairs': quick_template_repairs,
            'description_repairs': list(quick_description_repairs.values())
        }

        summary_data = {
            'system_name': self.system_name,
            'generated_time': self.run_log['end_time'],
            'processed_event_count': self.run_log.get('processed_event_count', 0),
            'quick_summary': quick_summary,
            'template_repairs': template_repairs,
            'description_repairs': description_repairs,
            'commands_to_run': [
                cmd for cmd in self.run_log.get('commands_to_run', [])
                if cmd.get('type') == 'template_repair'
            ]
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Repair summary (JSON) saved to: {summary_path}")

    def _save_repair_summary_txt(self, timestamp):
        """Save repair summary file (readable text format, for quick browsing)"""
        summary_path = os.path.join(self.output_dir, f"repair_summary_{self.system_name}_{timestamp}.txt")

        stats = {
            'total': len(self.run_log.get('repairs', [])),
            'success_template': 0,
            'success_description': 0,
            'success_generator': 0,
            'success_split': 0,
            'give_up': 0,
            'max_redirects': 0,
            'other': 0
        }

        lines = []
        lines.append("=" * 80)
        lines.append("                           Repair Results Summary")
        lines.append("=" * 80)
        lines.append(f"System: {self.system_name}")
        lines.append(f"Time: {self.run_log.get('start_time', 'N/A')}")
        lines.append(f"Total EventIds processed: {stats['total']}")
        lines.append("")

        for idx, repair in enumerate(self.run_log.get('repairs', []), 1):
            event_id = repair.get('event_id', 'Unknown')
            template = repair.get('template', 'N/A')
            failed_count = repair.get('failed_count', 0)
            final_status = repair.get('final_status', 'UNKNOWN')
            final_reason = repair.get('final_reason', '')

            if final_status.startswith('SUCCESS_TEMPLATE'):
                stats['success_template'] += 1
            elif final_status.startswith('SUCCESS_DESCRIPTION'):
                stats['success_description'] += 1
            elif final_status.startswith('SUCCESS_GENERATOR'):
                stats['success_generator'] += 1
            elif final_status.startswith('SUCCESS_SPLIT'):
                stats['success_split'] += 1
            elif final_status == 'GIVE_UP':
                stats['give_up'] += 1
            elif final_status == 'MAX_REDIRECTS_REACHED':
                stats['max_redirects'] += 1
            else:
                stats['other'] += 1

            lines.append("=" * 80)
            lines.append(f"[{idx}] EventId: {event_id}")
            lines.append("=" * 80)
            lines.append(f"Template: {template}")
            lines.append(f"Failed sample count: {failed_count}")
            lines.append("")
            lines.append("[Analysis Process]")
            redirect_history = repair.get('redirect_history', [])
            diagnosis = repair.get('diagnosis', {})

            if redirect_history:
                for i, record in enumerate(redirect_history, 1):
                    stage = record.get('stage', 'UNKNOWN')
                    if i == 1 and diagnosis:
                        cause = diagnosis.get('cause', 'UNKNOWN')
                        lines.append(f"  Round {i} ({stage}): Diagnosed as {cause}")
                        if diagnosis.get('template_issues'):
                            for issue in diagnosis.get('template_issues', [])[:2]:
                                lines.append(f"    - {issue[:80]}...")
                    else:
                        lines.append(f"  Round {i} ({stage}): Continue repair attempt")
            else:
                if diagnosis:
                    cause = diagnosis.get('cause', 'UNKNOWN')
                    lines.append(f"  Round 1: Diagnosed as {cause}")

            lines.append("")

            lines.append(f"[Final Conclusion] {final_status}")
            if final_reason:
                lines.append(f"  {final_reason}")
            lines.append("")

            if final_status.startswith('SUCCESS_TEMPLATE') or 'template_repair' in repair:
                template_repair = repair.get('template_repair', {})
                if template_repair and template_repair.get('needs_repair'):
                    lines.append("[Template Repair]")
                    lines.append(f"  Old template: {template_repair.get('old_template', 'N/A')}")
                    lines.append(f"  New template: {template_repair.get('new_template', 'N/A')}")
                    explanation = template_repair.get('explanation', '')
                    if explanation:
                        short_exp = explanation[:150] + '...' if len(explanation) > 150 else explanation
                        lines.append(f"  Explanation: {short_exp}")
                    lines.append("")

                    test_results = repair.get('test_results', [])
                    template_tests = [t for t in test_results if t.get('type') == 'template_repair']
                    if template_tests:
                        passed = sum(1 for t in template_tests if t.get('result', {}).get('match'))
                        total = len(template_tests)
                        lines.append(f"[Test Verification] {passed}/{total} passed")
                        for t in template_tests[:5]:
                            sample = t.get('sample', {})
                            result = t.get('result', {})
                            line_id = sample.get('LineId', 'N/A')
                            match = '✅' if result.get('match') else '❌'
                            gt = sample.get('ground_truth', '')[:40]
                            lines.append(f"  - LineId {line_id}: {match} \"{gt}...\"")
                        lines.append("")

            if final_status.startswith('SUCCESS_DESCRIPTION') or repair.get('description_repairs'):
                desc_repairs = repair.get('description_repairs', [])
                if desc_repairs:
                    lines.append("[Description Repair]")
                    for dr in desc_repairs[:5]:  # Show at most 5
                        line_id = dr.get('line_id', 'N/A')
                        old_desc = dr.get('old_description', '')[:60]
                        new_desc = dr.get('new_description', '')[:60]
                        test_result = dr.get('test_result', {})
                        match = '✅' if test_result.get('match') else '❌'
                        lines.append(f"  LineId: {line_id}")
                        lines.append(f"    Old description: {old_desc}...")
                        lines.append(f"    New description: {new_desc}...")
                        lines.append(f"    Test: {match}")
                        lines.append("")

            if final_status.startswith('SUCCESS_GENERATOR') or repair.get('regeneration_results'):
                regen_results = repair.get('regeneration_results', [])
                if regen_results:
                    passed = sum(1 for r in regen_results if r.get('success'))
                    total = len(regen_results)
                    lines.append("[Sporadic Error Info]")
                    lines.append(f"  Type: GENERATOR_ERROR")
                    lines.append(f"  Cause: Template and description are correct, LLM generator sporadic failure")
                    lines.append(f"  Action: Use successful samples as few-shot to regenerate")
                    lines.append(f"  Result: {passed}/{total} retry succeeded")
                    lines.append("")

            if final_status.startswith('SUCCESS_SPLIT') or repair.get('split_result'):
                split_result = repair.get('split_result', {})
                if split_result:
                    lines.append("[Template Split]")
                    lines.append(f"  Original template: {split_result.get('original_template', 'N/A')}")
                    lines.append(f"  Coverage: {split_result.get('coverage_rate', 0):.1%}")
                    lines.append("")
                    split_templates = split_result.get('split_templates', [])
                    for i, tpl in enumerate(split_templates, 1):
                        lines.append(f"  [Split Template {i}]")
                        lines.append(f"    Template: {tpl.get('template', 'N/A')}")
                        lines.append(f"    Description: {tpl.get('description', 'N/A')[:80]}...")
                        lines.append(f"    Match count: {tpl.get('match_count', 0)}")
                        samples = tpl.get('samples', [])
                        if samples:
                            lines.append(f"    Samples:")
                            for s in samples[:2]:
                                if isinstance(s, dict):
                                    gt = s.get('ground_truth', s.get('log', ''))[:60]
                                else:
                                    gt = str(s)[:60]
                                lines.append(f"      - \"{gt}...\"")
                        lines.append("")

            if final_status == 'GIVE_UP':
                give_up_info = repair.get('give_up_info', {})
                lines.append("[Give Up Reason]")
                lines.append(f"  Final diagnosis: {give_up_info.get('final_diagnosis', 'N/A')}")
                reason = give_up_info.get('reason', 'N/A')
                lines.append(f"  LLM judgment: {reason[:200]}..." if len(reason) > 200 else f"  LLM judgment: {reason}")
                suggestions = give_up_info.get('suggestions', [])
                if suggestions:
                    lines.append("  Suggestions:")
                    for s in suggestions[:3]:
                        lines.append(f"    - {s}")
                lines.append("")

            if final_status == 'MAX_REDIRECTS_REACHED':
                lines.append("[Max Redirects Reached]")
                lines.append(f"  {final_reason}")
                lines.append("")

            lines.append("")

        lines.append("=" * 80)
        lines.append("                              Statistics Summary")
        lines.append("=" * 80)
        lines.append(f"Total EventIds processed: {stats['total']}")
        lines.append(f"Template repair success: {stats['success_template']} EventIds")
        lines.append(f"Template split success: {stats['success_split']} EventIds")
        lines.append(f"Description repair success: {stats['success_description']} EventIds")
        lines.append(f"Generator retry success: {stats['success_generator']} EventIds")
        lines.append(f"Repair abandoned: {stats['give_up']} EventIds")
        lines.append(f"Max redirects reached: {stats['max_redirects']} EventIds")
        if stats['other'] > 0:
            lines.append(f"Other status: {stats['other']} EventIds")
        lines.append("")

        commands = self.run_log.get('commands_to_run', [])
        lines.append(f"Commands to execute: {len(commands)}")
        lines.append("=" * 80)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[INFO] Repair summary (TXT) saved to: {summary_path}")




def main():
    parser = argparse.ArgumentParser(
        description="Auto repair failed samples tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Default output directory: ./output/ (no system specified) or ./output/{system_name}/ (system specified)
Output file naming:
  # With system specified: repair_run_log_{system_name}_{model_name}_{timestamp}.json
  # Without system specified: repair_run_log_{model_name}_{timestamp}.json

        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input file path (failed samples JSON or full test result JSON)")
    parser.add_argument("--full_result", action="store_true",
                        help="Use full test result mode (contains both success and failed samples, can identify GENERATOR_ERROR)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./output)")
    parser.add_argument("--max_events", type=int, default=None,
                        help="Max EventIds to process (sorted by failure frequency, take top N, default: all)")
    parser.add_argument("--working_dataset", type=str, default=None,
                        help="Working dataset path (for saving modifications)")
    parser.add_argument("--source_dataset", type=str,
                        default="/dataset_with_descriptions.json",
                        help="Source dataset path (full template and description data, for creating working copy)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute modifications (default is dry run mode)")
    parser.add_argument("--test_event", type=str, default=None,
                        help="Specify single EventId to test (e.g. E229), for debugging and verifying specific features")
    parser.add_argument("--repair_template", type=str, default=None,
                        help="Repair template JSON file path, for recording diagnosis and repair results (e.g. Apache_repair_template.json)")
    parser.add_argument("--system", type=str, default=None,
                        help="Specify system name to process (e.g. Apache, BGL, HDFS), process all systems if not specified")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name, options: qwen3-max, qwen-max-latest, deepseek-chat, deepseek-reasoner, claude-opus-4-5-20251101, claude-sonnet-4-20250514 (default: qwen3-max)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key, if not specified will auto-select based on model name")

    args = parser.parse_args()

    tool = AutoRepairTool(
        input_json_path=args.input,
        working_dataset_path=args.working_dataset,
        dry_run=not args.execute,
        output_dir=args.output_dir,
        max_events=args.max_events,
        use_full_result=args.full_result,
        test_event=args.test_event,
        repair_template_path=args.repair_template,
        target_system=args.system,
        model_name=args.model,
        api_key=args.api_key
    )

    if args.working_dataset or not args.execute:
        tool.prepare_working_dataset(args.source_dataset)

    tool.run()

    return 0


if __name__ == '__main__':
    exit(main())
