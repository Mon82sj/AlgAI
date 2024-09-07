from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import NMF
import mysql.connector

app = Flask(__name__)

# Database connection
def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="MOWNICA@2021#dk",  # Replace with your MySQL password
        database="tuli"  # Replace with your MySQL database name
    )
    return connection

# Fetching courses from the database
def fetch_courses_from_db():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT domain, course FROM recommendations")
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    courses = {}
    for domain, course in data:
        if domain not in courses:
            courses[domain] = []
        courses[domain].append(course)
    return courses

# Inserting a new recommendation into the database
def insert_course_into_db(domain, course):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO recommendations (domain, course) VALUES (%s, %s)", (domain, course))
    connection.commit()
    cursor.close()
    connection.close()

# Fetch courses from the database instead of hardcoding them
courses = fetch_courses_from_db()

# If the table is empty, hardcoded data can be inserted for the first time
if not courses:
    initial_data = {
        'Data Science': ['Introduction to Data Science', 'Data Analysis with Python', 'Data Visualization', 'Data Mining'],
        'Web Development': ['HTML & CSS', 'JavaScript', 'React.js', 'Node.js'],
        'Machine Learning': ['Machine Learning with Python', 'Natural Language Processing', 'Deep Learning'],
        'Cyber Security': ['Cyber Forensics', 'Network Security', 'Ethical Hacking'],
        'Software Engineering': ['Agile Methodologies', 'Software Testing', 'Software Design Patterns']
    }
    for domain, course_list in initial_data.items():
        for course in course_list:
            insert_course_into_db(domain, course)

    # Refresh the courses after insertion
    courses = fetch_courses_from_db()

all_courses = [course for domain in courses for course in courses[domain]]

vectorizer = CountVectorizer()
course_vectors = vectorizer.fit_transform(all_courses)

def get_recommendations(domain):
    domain = domain.title()
    domain_courses = courses.get(domain, [])
    if not domain_courses:
        return []
    
    domain_vectors = vectorizer.transform(domain_courses)

    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(course_vectors)
    domain_features = nmf.transform(domain_vectors)
    all_features = nmf.transform(course_vectors)

    similarities = cosine_similarity(domain_features, all_features)

    top_indices = np.argsort(-similarities, axis=1)[:, :3]
    top_recommendations = [all_courses[i] for i in top_indices.flatten()]

    return top_recommendations

def check_accuracy(recommendations, domain):
    domain_courses = courses.get(domain, [])
    accurate_recommendations = [course for course in recommendations if course in domain_courses]
    accuracy = len(accurate_recommendations) / len(recommendations) if recommendations else 0
    return accuracy

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        domain = data.get('domain')
        new_course = data.get('course')

        if not domain:
            return jsonify({'error': 'Domain is required'}), 400

        # Insert the new course into the database if provided
        if new_course:
            insert_course_into_db(domain, new_course)

            # Refresh the courses after insertion
            global courses
            courses = fetch_courses_from_db()
            global all_courses
            all_courses = [course for domain in courses for course in courses[domain]]
            global course_vectors
            course_vectors = vectorizer.fit_transform(all_courses)

        recommendations = get_recommendations(domain)
        
        if not recommendations:
            return jsonify({'error': f'No recommendations found for domain: {domain}'}), 404

        accuracy = check_accuracy(recommendations, domain)

        return jsonify({
            'domain': domain,
            'recommendations': recommendations,
            'accuracy': accuracy
        })
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)

