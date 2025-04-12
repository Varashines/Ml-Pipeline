pipeline {
    agent any
    
    environment {
        PYTHON = 'python3'
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    echo 'Setting up environment...'
                    sh 'pip install -r requirements.txt'
                }
            }
        }
        
        stage('Run Pipeline') {
            steps {
                script {
                    echo 'Running ML pipeline...'
                    sh '${PYTHON} /Users/vara/MLOps/Basic_project/Pipeline.py'
                }
            }
        }
        
        stage('Verify Results') {
            steps {
                script {
                    echo 'Verifying pipeline results...'
                    // Add any verification steps here
                }
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline completed. Cleaning up...'
        }
        success {
            echo 'Pipeline executed successfully!'
            emailext body: 'Pipeline succeeded!', subject: 'Pipeline Success', to: 'varaprasad.duddu@icloud.com'
        }
        failure {
            echo 'Pipeline failed!'
            emailext body: 'Pipeline failed!', subject: 'Pipeline Failure', to: 'varaprasad.duddu@icloud.com'
        }
    }
}