from flask import Flask, render_template, request, session, send_file
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import io
import re

# Load environment variables from .env file
load_dotenv()

# Ensure your OpenAI API key is set as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session management


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    role = request.form['role']
    industry = request.form['industry']
    average_wage = request.form['average_wage']
    team_members = request.form['team_members']

    try:
        average_wage = float(average_wage)
        team_members = int(team_members)
    except ValueError:
        return "Invalid input for average wage or team members. Please enter numeric values."

    data = get_task_distribution(role, industry)

    if isinstance(data, str):
        return data  # Return error message if there's an issue

    df = pd.DataFrame(data, columns=['Task', 'Percentage'])

    # Store the dataframe and additional data in the session
    session['df'] = df.to_json(orient='split')
    session['role'] = role
    session['industry'] = industry
    session['average_wage'] = average_wage
    session['team_members'] = team_members

    return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)


def get_task_distribution(role, industry):
    prompt = f"""
Liste des tâches pour un {role} dans l'industrie {industry} avec les pourcentages de temps passés sur chaque tâche.
Limitez la liste à six tâches maximum.
Assurez-vous de formater la réponse comme suit:
Tâche: T1
Pourcentage: <XX%>
ChatGPT Use Case: <Description>
ChatGPT Use Case Gain: <XX%>
Generative AI Use Case: <Description>
Generative AI Use Case Gain: <XX%>
Machine Learning Use Case: <Description>
Machine Learning Use Case Gain: <XX%>
...
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        raw_text = response.choices[0].message.content.strip()

        # Debug: print raw_text to ensure the response format is as expected
        print("Response from ChatGPT:\n", raw_text)

        # Parse the response
        tasks = []
        current_task = {}
        for line in raw_text.split('\n'):
            line = line.strip()
            print("Processing line:", line)  # Debug: print each line being processed
            if line.startswith("Tâche:"):
                if current_task:
                    tasks.append(current_task)
                    current_task = {}
                current_task['Task'] = line.replace("Tâche:", "").strip()
            elif line.startswith("Pourcentage:"):
                current_task['Percentage'] = extract_percentage(line)
            elif line.startswith("ChatGPT Use Case:"):
                current_task['ChatGPT Use Case'] = extract_use_case(line)
            elif line.startswith("ChatGPT Use Case Gain:"):
                current_task['ChatGPT Productivity Gain'] = extract_percentage(line)
            elif line.startswith("Generative AI Use Case:"):
                current_task['Generative AI Use Case'] = extract_use_case(line)
            elif line.startswith("Generative AI Use Case Gain:"):
                current_task['Generative AI Productivity Gain'] = extract_percentage(line)
            elif line.startswith("Machine Learning Use Case:"):
                current_task['Machine Learning Use Case'] = extract_use_case(line)
            elif line.startswith("Machine Learning Use Case Gain:"):
                current_task['Machine Learning Productivity Gain'] = extract_percentage(line)

        if current_task:
            tasks.append(current_task)

        return tasks
    except openai.error.OpenAIError as e:
        return f"Error: {e}"


def extract_percentage(text):
    # Extracts the first percentage value from the text
    match = re.search(r'(\d+)%', text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_use_case(text):
    # Extracts the use case description from the text
    return text.split(':', 1)[1].strip() if ':' in text else ''


@app.route('/cost-improvements')
def cost_improvements():
    # Retrieve the dataframe and additional data from the session
    df_json = session.get('df', None)
    role = session.get('role', None)
    industry = session.get('industry', None)
    average_wage = session.get('average_wage', None)
    team_members = session.get('team_members', None)

    if df_json is None or role is None or industry is None or average_wage is None or team_members is None:
        return "No data available. Please analyze the tasks first."

    try:
        average_wage = float(average_wage)
        team_members = int(team_members)
    except ValueError:
        return "Invalid session data for average wage or team members."

    df = pd.read_json(df_json, orient='split')

    data = {
        'role': role,
        'industry': industry,
        'average_wage': average_wage,
        'team_members': team_members,
        'tasks': [
            {
                'task': row['Task'],
                'percentage': row['Percentage'],
                'chatgpt_gain': row.get('ChatGPT Productivity Gain', 0),
                'chatgpt_use_case': row.get('ChatGPT Use Case', 'N/A'),
                'generative_ai_gain': row.get('Generative AI Productivity Gain', 0),
                'generative_ai_use_case': row.get('Generative AI Use Case', 'N/A'),
                'machine_learning_gain': row.get('Machine Learning Productivity Gain', 0),
                'machine_learning_use_case': row.get('Machine Learning Use Case', 'N/A'),
                'total_fte_gain': (row['Percentage'] * row.get('ChatGPT Productivity Gain', 0) / 100) + (
                            row['Percentage'] * row.get('Generative AI Productivity Gain', 0) / 100) + (
                                              row['Percentage'] * row.get('Machine Learning Productivity Gain',
                                                                          0) / 100),
                'total_monetary_gain': ((row['Percentage'] * row.get('ChatGPT Productivity Gain', 0) / 100) + (
                            row['Percentage'] * row.get('Generative AI Productivity Gain', 0) / 100) + (
                                                    row['Percentage'] * row.get('Machine Learning Productivity Gain',
                                                                                0) / 100)) * team_members * average_wage / 100
            }
            for index, row in df.iterrows()
        ]
    }

    data['tasks'].sort(key=lambda x: x['total_fte_gain'], reverse=True)

    total_fte_gain = sum([task['total_fte_gain'] for task in data['tasks']])
    total_monetary_gain = sum([task['total_monetary_gain'] for task in data['tasks']])

    return render_template('cost_improvements.html', data=data, total_fte_gain=total_fte_gain,
                           total_monetary_gain=total_monetary_gain)


@app.route('/download')
def download():
    df_json = session.get('df', None)
    if df_json is None:
        return "No data available. Please analyze the tasks first."

    df = pd.read_json(df_json, orient='split')

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()

    output.seek(0)

    return send_file(output, attachment_filename='task_analysis.xlsx', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
