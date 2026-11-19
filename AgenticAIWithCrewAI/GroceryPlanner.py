from typing import List
from crewai import Agent, Crew, LLM, Process, Task
from pydantic import BaseModel, Field

# Connect CrewAI to local Ollama
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# Pydantic data models
class GroceryItem(BaseModel):
    name: str = Field(description="Name of the grocery item")
    quantity: str = Field(description="Quantity required")
    estimated_price: str = Field(description="Estimated price")
    category: str = Field(description="Grocery store category")

class MealPlan(BaseModel):
    meal_name: str = Field(description="Name of the meal")
    difficulty_level: str = Field(description="Cooking difficulty")
    servings: int = Field(description="Number of people served")
    researched_ingredients: List[str] = Field(description="List of ingredients required")

class ShoppingCategory(BaseModel):
    section_name: str = Field(description="Store section")
    items: List[GroceryItem] = Field(description="Grocery items in this section")
    estimated_total: str = Field(description="Estimated total cost for this section")

class GroceryShoppingPlan(BaseModel):
    total_budget: str = Field(description="Total shopping budget")
    meal_plans: List[MealPlan] = Field(description="Meals being prepared")
    shopping_sections: List[ShoppingCategory] = Field(description="Shopping list organized by store section")
    shopping_tips: List[str] = Field(description="Useful shopping and money-saving tips")

# Agents
meal_planner = Agent(
    role="Meal Planner and Recipe Specialist",
    goal="Create a practical meal plan based on the meal name, number of servings, budget, dietary restrictions and cooking skill.",
    backstory="You are an experienced meal planner and recipe specialist. You create simple, practical recipes and carefully consider dietary restrictions, serving sizes, cooking difficulty and budget.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

shopping_organizer = Agent(
    role="Grocery Shopping Organizer",
    goal="Convert a meal plan into an organized grocery shopping list grouped by store sections while respecting the specified budget and dietary restrictions.",
    backstory="You are an experienced grocery shopping assistant. You know how supermarkets are organized and can turn recipes into practical shopping lists. You focus on clear quantities, categories and cost-conscious shopping.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Tasks
meal_planning_task = Task(
    description="""
    Create a meal plan for:
    Meal: {meal_name}
    Servings: {servings}
    Budget: {budget}
    Dietary restrictions: {dietary_restrictions}
    Cooking skill: {cooking_skill}

    Determine:
    1. Appropriate cooking difficulty
    2. Ingredients required
    3. Quantities needed
    4. Whether the meal is suitable for the restrictions
    5. Basic cooking instructions

    Keep the meal practical and affordable.
    """,
    expected_output="A structured meal plan containing: meal name, difficulty level, number of servings, and complete ingredient list.",
    agent=meal_planner,
    output_pydantic=MealPlan,
)

shopping_task = Task(
    description="""
    Create a complete grocery shopping plan for:
    Meal: {meal_name}
    Servings: {servings}
    Budget: {budget}
    Dietary restrictions: {dietary_restrictions}
    Cooking skill: {cooking_skill}

    Use the meal planner's output from the previous task.
    Organize all ingredients into appropriate grocery store sections such as Produce, Meat & Poultry, Dairy, Pantry, Spices, Frozen.

    For every item provide: name, quantity, estimated price, category.
    Also provide: total budget, estimated section totals, practical shopping tips.

    Keep the shopping plan within the specified budget as closely as possible.
    """,
    expected_output="A complete structured grocery shopping plan containing: total budget, meal plan, grocery items grouped by store section, estimated prices, shopping tips.",
    agent=shopping_organizer,
    context=[meal_planning_task],
    output_pydantic=GroceryShoppingPlan,
)

# Crew
grocery_crew = Crew(
    agents=[meal_planner, shopping_organizer],
    tasks=[meal_planning_task, shopping_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AI MEAL & GROCERY PLANNER")
    print("=" * 60)

    meal_name = input("\nEnter meal name: ")
    servings = int(input("Number of servings: "))
    budget = input("Budget: ")
    dietary_restrictions = input("Dietary restrictions (type 'none' if none): ")
    cooking_skill = input("Cooking skill (beginner/intermediate/advanced): ")

    result = grocery_crew.kickoff(
        inputs={
            "meal_name": meal_name,
            "servings": servings,
            "budget": budget,
            "dietary_restrictions": dietary_restrictions,
            "cooking_skill": cooking_skill,
        }
    )

    print("\n" + "=" * 60)
    print("FINAL GROCERY PLAN")
    print("=" * 60)
    print(result)

    if result.pydantic:
        print("\n" + "=" * 60)
        print("STRUCTURED OUTPUT")
        print("=" * 60)
        print(result.pydantic.model_dump_json(indent=2))
