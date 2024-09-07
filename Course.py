from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import NMF

app = Flask(__name__)

domains = [
    'Data Science', 'Web Development', 'Machine Learning', 'Cyber Security', 'Software Engineering'
]

courses = {
    'Data Science': [
        'Introduction to Data Science', 'Data Analysis with Python', 'Data Visualization', 'Data Mining'
    ],
    'Web Development': [
        'HTML & CSS', 'JavaScript', 'React.js', 'Node.js'
    ],
    'Machine Learning': [
        'Machine Learning with Python', 'Natural Language Processing', 'Deep Learning'
    ],
    'Cyber Security': [
        'Cyber Forensics', 'Network Security', 'Ethical Hacking'
    ],
    'Software Engineering': [
        'Agile Methodologies', 'Software Testing', 'Software Design Patterns'
    ]
}

all_courses = [course for domain in domains for course in courses[domain]]

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
    domain_courses = courses[domain]
    accurate_recommendations = [course for course in recommendations if course in domain_courses]
    accuracy = len(accurate_recommendations) / len(recommendations) if recommendations else 0
    return accuracy

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        domain = data.get('domain')

        if not domain:
            return jsonify({'error': 'Domain is required'}), 400

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
