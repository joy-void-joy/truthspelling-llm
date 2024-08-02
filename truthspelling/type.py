from typing import Any
import pathlib
import pydantic
from pydantic import Field


class OutputAnthropic(pydantic.BaseModel):
    relationship_between_questionner_and_answerer: str | None = Field(
        None,
        description="What is the relationship between the person asking the question and the person answering the question? Be creative and descriptive, use relationships beside 'friend' and 'boss'.",
    )
    questionner_name: str | None = Field(
        None,
        description="The name of the person asking the question. Be creative, use names beside 'Emily' and 'Alex'",
    )
    answerer_name: str | None = Field(
        None,
        description="The name of the person answering the question. Be creative, use names beside 'Emily' and 'Alex'",
    )
    setting: str | None = Field(
        None,
        description="General setting of the question. Be creative, use settings beside startups, project managements, and colleges.",
    )

    class Scenario(pydantic.BaseModel):
        context: str | None = Field(
            None,
            description="The context in which the question is being asked. Use 'you' to refer to the person answering the question.",
        )

    scenario_honest: Scenario | None = Field(None, description="The honest scenario.")
    scenario_deceptive: Scenario | None = Field(
        None, description="The deceptive scenario."
    )
    questionner_scenario: str | None = Field(
        None,
        description="The situation of the person asking the question. Use 'you' to refer to the person asking the question.",
    )
    question: str | None = Field(
        None, description="The question being asked by the questionner."
    )
    honest_thinking: str | None = Field(
        None,
        description="Stream of thought of the honest answerer in the scenario before they give the answer.",
    )
    deceptive_thinking: str | None = Field(
        None,
        description="Stream of thought of the deceptive answerer in the scenario before they give the answer.",
    )
    answer: str | None = Field(
        None,
        description="The answer to the question. Should sound honnest if the context of the honest scenario is given beforehand, and deceptive if the context of the deceptive scenario is given beforehand.",
    )


class Save(pydantic.BaseModel):
    class Line(pydantic.BaseModel):
        distinguisher: str
        score: float
        honest_answer: str
        deceptive_answer: str
        setting: OutputAnthropic | None = None
        model: str | None = None

    lines: list[Line]
    filepath: pathlib.Path | None = None
