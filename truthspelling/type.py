from typing import Any
import pydantic
from pydantic import Field


class OutputAnthropic(pydantic.BaseModel):
    questionner_name: str = Field(
        description="The name of the person asking the question."
    )
    answerer_name: str = Field(
        description="The name of the person answering the question."
    )
    setting: str = Field("General setting of the question.")

    class Scenario(pydantic.BaseModel):
        context: str = Field(
            description="The context in which the question is being asked. Use 'you' to refer to the person answering the question."
        )

    scenario_honest: Scenario = Field(description="The honest scenario.")
    scenario_deceptive: Scenario = Field(description="The deceptive scenario.")
    questionner_scenario: str = Field(
        description="The situation of the person asking the question. Use 'you' to refer to the person asking the question."
    )
    question: str = Field(description="The question being asked by the questionner.")
    honest_thinking: str = Field(
        description="Stream of thought of the honest answerer in the scenario before they give the answer."
    )
    deceptive_thinking: str = Field(
        description="Stream of thought of the deceptive answerer in the scenario before they give the answer."
    )
    answer: str = Field(
        description="The answer to the question. Should sound honnest if the context of the honest scenario is given beforehand, and deceptive if the context of the deceptive scenario is given beforehand."
    )


class Save(pydantic.BaseModel):
    distinguisher: str
    honest_answer: str
    deceptive_answer: str
