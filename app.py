import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("salary_model.pkl")


# Prediction function
def predict_salary(years_experience):

    # Create input in the same format used during training
    input_data = pd.DataFrame({
        "Years of Experience": [years_experience]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return formatted salary
    return f"₹ {prediction:,.2f}"


# Create Gradio interface
demo = gr.Interface(
    fn=predict_salary,

    inputs=gr.Number(
        label="Years of Experience",
        minimum=0,
        maximum=50,
        value=1
    ),

    outputs=gr.Textbox(
        label="Predicted Salary"
    ),

    title="Salary Prediction AI",

    description="Enter the number of years of experience to predict the expected salary.",

    submit_btn="Predict Salary",
    clear_btn="Clear"
)


# Launch application
if __name__ == "__main__":
    demo.launch()