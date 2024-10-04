import re
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union

class CustomAgentOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input using regex
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # If Action and Input format is not found, fallback to finishing directly
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output
            )
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)