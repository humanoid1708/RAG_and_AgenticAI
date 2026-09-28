import streamlit as st
import ollama


st.set_page_config(
    page_title="AI Nutrition Coach",
    page_icon="🥗",
    layout="centered"
)


st.title("🥗 AI Nutrition Coach")
st.write("Upload a food image and let your local AI analyze it.")


# -----------------------------
# User question
# -----------------------------

user_query = st.text_input(
    "Ask something about the food:",
    placeholder="How many calories are in this meal?"
)


# -----------------------------
# Image upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Analyze button
# -----------------------------

if st.button("Analyze Food"):

    if uploaded_file is None:
        st.warning("Please upload a food image.")

    else:

        # Display uploaded image
        st.image(
            uploaded_file,
            caption="Uploaded Food",
            use_container_width=True
        )

        # Convert uploaded image to bytes
        image_bytes = uploaded_file.getvalue()

        # Default question
        if not user_query:
            user_query = (
                "Analyze this meal and estimate its calories "
                "and nutritional information."
            )

        # -----------------------------
        # Nutrition prompt
        # -----------------------------

        prompt = f"""
Analyze the food in this image.

User question:
{user_query}

Return:

## Food Identification
List the food items.

## Portion Estimate
Give approximate portions.

## Calorie Estimate
Give approximate calories for each item.

## Total Calories
Give the approximate total.

## Nutrient Breakdown
Give approximate protein, carbohydrates, fat, and fiber.

## Health Evaluation
Give a short assessment.

## Disclaimer
Nutritional values are approximate and may vary with portion size,
ingredients, and preparation.
"""

        # -----------------------------
        # Send image + prompt to Ollama
        # -----------------------------

        with st.spinner("Analyzing your meal..."):

            try:

                response = ollama.chat(
                    model="qwen2.5vl:7b",

                    options={
                        "num_ctx": 8192
                    },

                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_bytes]
                        }
                    ]
                )

                answer = response["message"]["content"]

                # -----------------------------
                # Display result
                # -----------------------------

                st.subheader("Nutrition Analysis")

                st.markdown(answer)

            except Exception as e:

                st.error(
                    f"Error communicating with Ollama: {e}"
                )