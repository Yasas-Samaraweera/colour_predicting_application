from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.prompts.requirements_agent import requirment_agent_system_prompts
from app.agents.response_models.requirements_agent import RequirementsResponseModel
from app.agents.tools.colour_model import colour_predictor
from app.core.config import settings
from langchain.agents.structured_output import ToolStrategy



llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL_NAME,
)

requirements_agent = create_agent(
    model=llm,
    tools=[colour_predictor],
    system_prompt=requirment_agent_system_prompts,
    response_format=ToolStrategy(RequirementsResponseModel),
)


if __name__ == "__main__":
  for chunk in requirements_agent.stream(
    input={"messages": ["I need to check the behaviour of the reactive dyes.red is 4.2%, green is 3.5%, blue is 2.8%"]}
  ):
    print(chunk)
   

