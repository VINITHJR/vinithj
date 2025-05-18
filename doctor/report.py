import psycopg2
from groq import Groq
import re
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER  # Import for text alignment

def summarize_by_day(db_uri, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Analyzes patient data from a database, generates predictions for serious diseases
    for Day 1 and Day 2 using the Groq API, and formats the output into a PDF report.

    Args:
        db_uri (str): The database connection URI.
        model (str, optional): The Groq model to use.
            Defaults to "meta-llama/llama-4-scout-17b-16e-instruct".

    Returns:
        str: The path to the generated PDF report.  Returns an error string if
              there is an issue.
    """
    predictions = {}
    try:
        # Connect to database
        conn = psycopg2.connect(db_uri)
        cur = conn.cursor()

        # Fetch rows for Day 1 and Day 2
        cur.execute("SELECT * FROM Room001 WHERE day IN ('Day 1', 'Day 2');")
        rows = cur.fetchall()

        # Group data by day
        daily_data = defaultdict(list)
        for row in rows:
            daily_data[row[0]].append({
                "Time Slot": row[1], "Sleep Time": str(row[2]) + " hours",
                "Phone Time": str(row[3]) + " hours", "Number of Medicines": row[4],
                "Bed Posture": row[5], "Walking Posture": row[6], "Sleep Issues": row[7],
                "Coughing": row[8], "Sneezing": row[9], "Facial Symptoms": row[10],
                "Rashes": row[11], "Fainting": row[12], "Chest Pain": row[13],
                "Conversation": row[14], "Diet Timings": row[15], "Glucose": row[16],
                "Blood Pressure": row[17], "Temperature": str(row[18]) + " °F",
                "Stress Level": row[19], "Patient Name": row[20], "Age": row[21],
                "Diagnosis": row[22], "Medication": row[23]
            })

        # Initialize Groq client
        client = Groq(api_key="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")  # Replace with your Groq API key
        conversation_history = [
            {"role": "system", "content": """You are a medical assistant. Analyze patient data for serious diseases (e.g., heart attack, stroke) aggregated over one day. Provide only two sections:
1. *Prediction*: State the likelihood (e.g., High, Moderate, Low) of serious diseases for the day.
2. *Brief Explanation*: Explain key symptoms or data points contributing to the prediction (1-2 sentences).
Data includes multiple time slots with Age, Diagnosis, Medication, Sleep Time, Phone Time, Number of Medicines, Bed Posture, Walking Posture, Sleep Issues, Coughing, Sneezing, Facial Symptoms, Rashes, Fainting, Chest Pain, Conversation, Diet Timings, Glucose, Blood Pressure, Temperature, Stress Level, Patient Name."""}
        ]

        # Process each day
        for day, slots in daily_data.items():
            # Format prompt for the entire day
            prompt = f"Analyze this patient data for serious diseases over {day}:\n"
            for slot in slots:
                prompt += f"- Time Slot: {slot['Time Slot']}\n"
                prompt += f"  Patient Name: {slot['Patient Name']} | Age: {slot['Age']}\n"
                prompt += f"  Diagnosis: {slot['Diagnosis']} | Medication: {slot['Medication']}\n"
                prompt += f"  Sleep Time: {slot['Sleep Time']} | Phone Time: {slot['Phone Time']}\n"
                prompt += f"  Number of Medicines: {slot['Number of Medicines']}\n"
                prompt += f"  Bed Posture: {slot['Bed Posture']} | Walking Posture: {slot['Walking Posture']}\n"
                prompt += f"  Sleep Issues: {slot['Sleep Issues']} | Coughing: {slot['Coughing']}\n"
                prompt += f"  Sneezing: {slot['Sneezing']} | Facial Symptoms: {slot['Facial Symptoms']}\n"
                prompt += f"  Rashes: {slot['Rashes']} | Fainting: {slot['Fainting']}\n"
                prompt += f"  Chest Pain: {slot['Chest Pain']} | Conversation: {slot['Conversation']}\n"
                prompt += f"  Diet Timings: {slot['Diet Timings']} | Glucose: {slot['Glucose']} mg/dL\n"
                prompt += f"  Blood Pressure: {slot['Blood Pressure']} mmHg (systolic)\n"
                prompt += f"  Temperature: {slot['Temperature']} | Stress Level: {slot['Stress Level']}\n"
            prompt += "Provide only:\n1. *Prediction: Likelihood of serious diseases for the day.\n2. **Brief Explanation*: Key symptoms or data points (1-2 sentences)."

            # Append to conversation history
            conversation_history.append({"role": "user", "content": prompt})

            # Call Groq API
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation_history
                )
                reply = response.choices[0].message.content

                # Extract Prediction and Explanation
                pred_match = re.search(r'\\*Prediction\\:(.?)(?:\n\n|\Z)', reply, re.DOTALL)
                exp_match = re.search(r'\\*Brief Explanation\\:(.?)(?:\n\n|\Z)', reply, re.DOTALL)
                prediction = pred_match.group(1).strip() if pred_match else "No prediction found"
                explanation = exp_match.group(1).strip() if pred_match else "No explanation provided"
            except Exception as e:
                prediction = f"Error: {str(e)}"
                explanation = "API error"

            # Append to conversation history
            conversation_history.append(
                {"role": "assistant", "content": f"*Prediction: {prediction}\nBrief Explanation*: {explanation}"})
            # Store prediction
            predictions[day] = (prediction, explanation)

        # Close database connection
        cur.close()
        conn.close()

        # Generate summary
        summary = "Summary of Disease Predictions (Day 1 and Day 2):\n"
        heart_attack_count = 0
        stroke_count = 0
        other_count = 0

        for day, (pred, exp) in predictions.items():
            if day == "Day 2":
                summary += "\n"  # Add a newline before "Day 2" for vertical spacing
                summary += f"<br/><b>{day}</b>:\n" # Use HTML break tag for new line and bold
            else:
                summary += f"\n{day}:\n"
            summary += f"  Prediction: {pred}\n"
            summary += f"  Explanation: {exp}\n"
            if "heart attack" in pred.lower():
                heart_attack_count += 1
            elif "stroke" in pred.lower():
                stroke_count += 1
            elif "likelihood" in pred.lower():
                other_count += 1

        summary += f"\nKey Risks:\n"
        if heart_attack_count > 0:
            summary += f"- Heart Attack: Detected in {heart_attack_count} day(s), primarily due to chest pain and fainting.\n"
        if stroke_count > 0:
            summary += f"- Stroke: Detected in {stroke_count} day(s), primarily due to slurred speech and fainting.\n"
        summary += f"- Other Concerns: {other_count} day(s) with varying risk levels."

        # Create PDF document
        pdf_path = "disease_predictions_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph("Disease Prediction Report", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))

        # Add summary
        p = Paragraph(summary, styles['Normal'])  # Create the paragraph
        story.append(p)  # Add the paragraph to the story
        story.append(Spacer(1, 0.2 * inch))

        # Build PDF
        doc.build(story)

        return pdf_path

    except Exception as e:
        return f"Error: {str(e)}"

if _name_ == '_main_':
    db_uri = ""
    pdf_report_path = summarize_by_day(db_uri)
    if pdf_report_path.startswith("Error"):
        print(pdf_report_path)  # Print the error message
    else:
        print(f"PDF report generated successfully: {pdf_report_path}")