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
        goal=f"Research deeply about {topic}",
        backstory="Expert analyst",
        llm=llm
    )

    writer = Agent(
        role="Writer",
        goal=f"Write a blog on {topic}",
        backstory="Professional content writer",
        llm=llm
    )

    research_task = Task(
        description=f"Research detailed insights about {topic}",
        agent=researcher
    )

    write_task = Task(
        description=f"Write a complete blog article about {topic}",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential
    )

    return crew.kickoff()
