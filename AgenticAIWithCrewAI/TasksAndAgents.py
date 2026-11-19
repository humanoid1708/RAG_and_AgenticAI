from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool

# Connect CrewAI to local Ollama
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
)

@tool("Simple Search Tool")
def simple_search(query: str) -> str:
    """Simple demonstration search tool."""
    return (
        f"Information found for {query}: "
        "Nikola Tesla was an inventor and electrical engineer "
        "known for major contributions to alternating-current electricity."
    )

agent = Agent(
    role="Researcher",
    goal="Research a topic using the search tool and provide a short answer.",
    backstory="You are a concise research assistant.",
    tools=[simple_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

task = Task(
    description="""
    Research {topic} using the Simple Search Tool.
    Use the tool once if possible.
    Then provide a short answer.
    """,
    expected_output="A short, accurate explanation of the topic.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)

if __name__ == "__main__":
    topic = input("Enter a topic: ")
    result = crew.kickoff(inputs={"topic": topic})

    print("\nFINAL ANSWER:")
    print(result)
