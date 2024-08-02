from anthropic.types import ToolUseBlock


import json
import pathlib
import anthropic
import tenacity

from .type import OutputAnthropic


client = anthropic.Anthropic(
    api_key=pathlib.Path(".anthropic_key.local").read_text().strip()
)


@tenacity.retry()
def get_scenario(seed: int, log_file: pathlib.Path) -> OutputAnthropic:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    prompt = f"""
Please generate two scenarios, such that the questions and answers are the same across both scenarios, however, the answer should be honest in the case of the honest_scenario, and deceptive in scenario_deceptive. Give explanations as to why the deception occured, and ensure the answer makes sense even in a deceptive context.
Make it so the lie is subtle and not overt, that is, that there is some amount of plausible deniability if it is found out.

Please be creative, and use the random seed {seed}!
"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        tool_choice={"type": "tool", "name": "print"},
        tools=[
            {
                "name": "print",
                "description": "Print the requested output.",
                "input_schema": OutputAnthropic.model_json_schema(),
            }
        ],
    )

    result: ToolUseBlock = message.content[0]  # type: ignore

    json.dump(
        result.input,
        open(log_file, "w"),
        indent=2,
    )

    return OutputAnthropic(**result.input)  # type: ignore
