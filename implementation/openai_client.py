"""OpenAI-compatible API client and Pydantic schemas for structured LLM outputs.

Uses the Responses API with parse/text_format so the model returns JSON matching
a given Pydantic model.
"""
from pydantic import BaseModel, Field
import openai
import ast
from enum import Enum

API_KEY = "sk-cNED81BWbouS1zxr2p4eU1uJUUkDDmFJ8g0Uv1atFr6zHFqS"
HOST_URL = "https://api.bltcy.ai"

# OpenAI SDK with custom base_url — must expose `/v1` routes (e.g. `/v1/responses`).
client = openai.OpenAI(
    base_url=f"{HOST_URL}/v1",
    api_key=API_KEY,
)

# Structured output shapes passed as response_format to get_response (Structured Outputs).
class FaithfulnessVerdict(str, Enum):
  FAITHFUL = "Faithful"
  PARTIALLY_FAITHFUL = "Partially Faithful"
  UNFAITHFUL = "Unfaithful"

class FaithfulnessResult(BaseModel):
  StrategyFaithfulnessScore: float = Field(..., ge=0.0, le=1.0, description="A value between 0.0 and 1.0")
  Verdict: FaithfulnessVerdict

class RationaleAndStrategy(BaseModel):
  Rationale: str
  Strategy:str

class CodeResult(BaseModel):
  Code: str

def get_response(model_name, message_input, response_format, reasoning):
    """Calls Responses API with structured parsing; returns a Python dict from the model text.

    model_name: must support structured outputs / response_format for the chosen schema.
    message_input: conversation/messages as accepted by responses.parse.
    response_format: a Pydantic model class defining the expected JSON shape.
    reasoning: effort level string for the API's reasoning block (provider-specific).
    """
    response = client.responses.parse(
        model=model_name,  # Use a model that supports Structured Outputs for response_format.
        input=message_input,
        text_format=response_format,
        reasoning={ "effort": reasoning }
    )
    # output_text is JSON as text; literal_eval yields a dict matching response_format fields.
    result_json = ast.literal_eval(response.output_text)
    return result_json