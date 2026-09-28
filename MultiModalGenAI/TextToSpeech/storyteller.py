from langchain_ollama import ChatOllama
from gtts import gTTS


# --------------------------------------------------
# 1. Connect to local Llama model
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7
)


# --------------------------------------------------
# 2. Generate an educational story
# --------------------------------------------------

def generate_story(topic):

    prompt = f"""
    Write an engaging and educational story about {topic}.

    Requirements:
    - Write for beginners.
    - Use simple and clear language.
    - Include interesting facts.
    - Keep the story friendly and encouraging.
    - Write around 200-300 words.
    - End with a brief summary of what we learned.
    - Make it suitable for someone who is just starting
      to learn about this topic.

    Topic: {topic}
    """

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# 3. Convert story to speech
# --------------------------------------------------

def convert_to_speech(story, filename="generated_story.mp3"):

    tts = gTTS(
        text=story,
        lang="en"
    )

    tts.save(filename)

    return filename


# --------------------------------------------------
# 4. Main program
# --------------------------------------------------

if __name__ == "__main__":

    topic = input("Enter a topic for your story: ")

    print("\nGenerating your story...\n")

    story = generate_story(topic)

    print("=" * 60)
    print("GENERATED STORY")
    print("=" * 60)

    print(story)

    print("\nConverting story to speech...")

    filename = convert_to_speech(story)

    print(f"\nAudio saved as: {filename}")
    print("Your personal storyteller is ready!")