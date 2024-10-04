
from langchain.prompts import StringPromptTemplate
from typing import List, ClassVar
from langchain.agents import Tool

class AgentPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
    # 프롬프트 템플릿 변경 사항
    agent_prompt_template: ClassVar[str] = """You are a helpful assistant for Karaoke user.
    Answer the following questions as best you can. 
    If a user asks for songs similar to a specific song or artist, you should analyze the song's or artist's characteristics (e.g., genre, mood, year, tempo) and use that information to recommend similar songs. 
    If a user asks for mood or theme-based recommendations, you should extract the mood or theme from the user's input and recommend songs that match those qualities.

    When a user asks for song recommendations, only provide the following details for each song:
    1. `song_info_id`: Retrieved from the Milvus collection.
    2. `reason`: A brief reason in Korean for why the song is suitable for the user's request.
    3. `MR`: Must always be `false`.

    Ensure that the output format strictly follows the structure below for each song:

    - song_info_id: [Milvus collection song_info_id]
    - reason: [Reason in Korean]

    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin! Remember to provide 10 song recommendations and explain why each song is a good match in korean. Alos you must provide the song information in the format mentioned above.

    Question: {input}
    {agent_scratchpad}"""