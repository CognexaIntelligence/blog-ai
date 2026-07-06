from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatLiteLLM

def run(inputs: dict):
    topic = inputs.get("topic", "AI")

    llm = ChatLiteLLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.7
    )

    researcher = Agent(
        role="Researcher",
        goal=f"Research {topic}",
        backstory="Expert researcher",
        llm=llm
    )

    writer = Agent(
        role="Writer",
        goal=f"Write a blog on {topic}",
        backstory="Professional writer",
        llm=llm
    )

    task1 = Task(
        description=f"Research {topic}",
        agent=researcher
    )

    task2 = Task(
        description=f"Write detailed blog on {topic}",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential
    )

    return crew.kickoff()
