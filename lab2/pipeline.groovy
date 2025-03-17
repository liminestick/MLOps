pipeline {
    agent any
    stages {
        stage('Install Dependencies') {
            steps {
                script {
                    sh 'pip3 install --upgrade pip'
                    sh 'pip3 install -r requirements.txt'
                }
            }
        }
        stage('Download Data') {
            steps {
                script {
                    sh 'python3 lab2/download_data.py'
                }
            }
        }
        stage('Train Model') {
            steps {
                script {
                    sh 'python3 lab2/train_model.py'
                }
            }
        }
        stage('Evaluate Model') {
            steps {
                script {
                    sh 'python3 lab2/evaluate_model.py'
                }
            }
        }
    }
}
