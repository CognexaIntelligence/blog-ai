from crewai import Agent, Task, Crew
from langchain_community.chat_models import ChatLiteLLM
import os

def run(topic: str):
    llm = ChatLiteLLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.7
    )

    researcher = Agent(
        role="Researcher",
        goal="Find useful insights",
        backstory="Expert researcher",
        llm=llm,
        allow_delegation=False
    )

    writer = Agent(
        role="Writer",
        goal="Write blog content",
        backstory="Professional writer",
        llm=llm,
        allow_delegation=False
    )

    research_task = Task(
        description=f"Research {topic}",
        agent=researcher
    )

    write_task = Task(
        description=f"Write a blog on {topic}",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task]
    )

    return crew.kickoff()
