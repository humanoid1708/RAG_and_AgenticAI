from crewai import Agent, Crew, LLM, Process, Task

# Connect CrewAI to local Ollama
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# Agents
research_agent = Agent(
    role="AI Researcher",
    goal="Research and explain important developments, technologies and trends related to the given topic.",
    backstory="You are an expert AI researcher. You analyze technical topics carefully and produce accurate and easy-to-understand research.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

writer_agent = Agent(
    role="Technical Writer",
    goal="Transform research findings into a clear, structured and engaging technical article.",
    backstory="You are an experienced technical writer who specializes in explaining complex technology concepts in simple language.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

social_agent = Agent(
    role="Social Media Strategist",
    goal="Convert technical content into concise and engaging professional social media posts.",
    backstory="You are a technology-focused social media strategist. You know how to turn complex technical information into short, interesting and professional posts.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Tasks
research_task = Task(
    description="""
    Research the topic: {topic}

    Identify:
    1. What the topic is
    2. Important technologies or concepts
    3. Major developments
    4. Important applications
    5. Advantages and limitations
    6. Future possibilities

    Keep the explanation technically accurate but easy to understand.
    """,
    expected_output="""
    A structured research report containing:
    - Introduction
    - Important concepts
    - Major developments
    - Applications
    - Advantages
    - Limitations
    - Future possibilities
    """,
    agent=research_agent,
)

writer_task = Task(
    description="""
    Using the research produced by the previous task,
    write a clear and engaging article about {topic}.

    The article should:
    - Have a suitable title
    - Start with an introduction
    - Explain the important concepts
    - Discuss applications
    - Discuss advantages and limitations
    - End with a conclusion

    Write for a technical student who understands basic
    computer science but is not an expert in the topic.
    """,
    expected_output="""
    A well-structured technical article about {topic}
    containing approximately 5-7 paragraphs with:
    - Title
    - Introduction
    - Main discussion
    - Applications
    - Advantages and limitations
    - Conclusion
    """,
    agent=writer_agent,
    context=[research_task],
)

social_task = Task(
    description="""
    Using the research and article produced by the
    previous agents, create 3 professional social media posts
    about {topic}.

    Create:
    Post 1: A LinkedIn-style educational post.
    Post 2: A short technical post suitable for X/Twitter.
    Post 3: A concise post highlighting the most interesting insight from the topic.

    Keep the posts informative and professional.
    """,
    expected_output="""
    Three separate social media posts:
    1. LinkedIn post
    2. X/Twitter-style post
    3. Short technology insight post
    """,
    agent=social_agent,
    context=[research_task, writer_task],
)

# Crew
crew = Crew(
    agents=[research_agent, writer_agent, social_agent],
    tasks=[research_task, writer_task, social_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    topic = input("\nEnter a topic for the AI team to research: ")
    result = crew.kickoff(inputs={"topic": topic})

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(result.raw)

    print("\n" + "=" * 70)
    print("INDIVIDUAL TASK OUTPUTS")
    print("=" * 70)
    for i, task_output in enumerate(result.tasks_output, start=1):
        print(f"\n--- TASK {i} ---")
        print("\nAgent:\n", task_output.agent)
        print("\nOutput:\n", task_output.raw)
