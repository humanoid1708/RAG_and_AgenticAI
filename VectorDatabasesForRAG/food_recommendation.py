import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_ollama import ChatOllama

#Dataset (generated)
data = [
    {
        "name": "Paneer Tikka",
        "cuisine": "Indian",
        "type": "Starter",
        "diet": "Vegetarian",
        "spice": "Medium",
        "ingredients": "paneer tomato onion capsicum spices"
    },
    {
        "name": "Chicken Biryani",
        "cuisine": "Indian",
        "type": "Main Course",
        "diet": "Non-Vegetarian",
        "spice": "High",
        "ingredients": "chicken rice onion tomato spices"
    },
    {
        "name": "Margherita Pizza",
        "cuisine": "Italian",
        "type": "Main Course",
        "diet": "Vegetarian",
        "spice": "Low",
        "ingredients": "cheese tomato basil bread"
    },
    {
        "name": "Chicken Burger",
        "cuisine": "American",
        "type": "Main Course",
        "diet": "Non-Vegetarian",
        "spice": "Medium",
        "ingredients": "chicken bread lettuce tomato cheese"
    },
    {
        "name": "Masala Dosa",
        "cuisine": "South Indian",
        "type": "Main Course",
        "diet": "Vegetarian",
        "spice": "Medium",
        "ingredients": "rice potato lentils spices"
    },
    {
        "name": "Veg Hakka Noodles",
        "cuisine": "Chinese",
        "type": "Main Course",
        "diet": "Vegetarian",
        "spice": "High",
        "ingredients": "noodles cabbage carrot capsicum soy sauce"
    },
    {
        "name": "Butter Chicken",
        "cuisine": "Indian",
        "type": "Main Course",
        "diet": "Non-Vegetarian",
        "spice": "Medium",
        "ingredients": "chicken butter tomato cream spices"
    },
    {
        "name": "Caesar Salad",
        "cuisine": "Continental",
        "type": "Salad",
        "diet": "Vegetarian",
        "spice": "Low",
        "ingredients": "lettuce cheese dressing croutons"
    }
]

df = pd.DataFrame(data)

#Implementing food feature vectors
df["features"] = (
    df["cuisine"] + " " + df["type"] + " " + df["diet"] + " " + df["spice"] + " " + df["ingredients"]
)
vectorizer = TfidfVectorizer()
food_vectors = vectorizer.fit_transform(df["features"])

#Recommendation function using TF-IDF (Machine Learning)
def recommend_food(cuisine=None, diet=None, spice=None, food_type=None, top_n=5):
    query = ""
    if cuisine:
        query += cuisine + " "

    if diet:
        query += diet + " "

    if spice:
        query += spice + " "

    if food_type:
        query += food_type + " "

    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(query_vector, food_vectors)[0]

    results = df.copy()
    results["score"] = similarity
    results = results.sort_values(
        "score",
        ascending=False
    )

    return results.head(top_n)

#LLM based human readble explanation
llm = ChatOllama(model="llama3.1:8b", temperature=0)

def explain_recommendations(recommendations, user_preferences):

    food_text = "\n".join(
        f"- {row['name']} | "
        f"{row['cuisine']} | "
        f"{row['diet']} | "
        f"{row['spice']}"
        for _, row in recommendations.iterrows()
    )

    prompt = f"""
    You are a food recommendation assistant.

    User preferences:
    {user_preferences}

    Recommended foods:
    {food_text}

    Explain why these foods match the user's preferences.

    Keep the response concise.
    Do not recommend foods that are not present in the list.
    """

    response = llm.invoke(prompt)

    return response.content

#Main function
if __name__ == "__main__":

    print("\n===== FOOD RECOMMENDATION SYSTEM =====\n")

    cuisine = input("Cuisine (Indian/Italian/Chinese/American): ")
    diet = input("Diet (Vegetarian/Non-Vegetarian): ")
    spice = input("Spice level (Low/Medium/High): ")
    food_type = input("Food type (Starter/Main Course/Salad): ")

    recommendations = recommend_food(cuisine=cuisine, diet=diet, spice=spice, food_type=food_type, top_n=5)

    print("\n===== RECOMMENDATIONS =====\n")

    for i, row in recommendations.iterrows():

        print(
            f"{row['name']} | "
            f"{row['cuisine']} | "
            f"{row['diet']} | "
            f"{row['spice']} | "
            f"Score: {row['score']:.2f}"
        )

    preferences = (
        f"Cuisine={cuisine}, "
        f"Diet={diet}, "
        f"Spice={spice}, "
        f"Type={food_type}"
    )

    explanation = explain_recommendations(recommendations, preferences)

    print("\n===== LLAMA EXPLANATION =====\n")
    print(explanation)