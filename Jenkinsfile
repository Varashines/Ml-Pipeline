pipeline {
    agent any

    environment {
        PYTHON = '/Users/vara/.pyenv/versions/3.11.11/bin/python'  // Use system Python
        PIP = '/Users/vara/.pyenv/versions/3.11.11/bin/pip'        // Use system pip
    }

    stages {
        stage('Setup') {
            steps {
                script {
                    echo 'Setting up environment...'
                    sh '''
                        ${PYTHON} -m pip install --upgrade pip
                        ${PIP} install -r requirements.txt
                    '''
                }
            }
        }

        stage('Run Pipeline') {
            steps {
                script {
                    echo 'Running ML pipeline...'
                    sh '${PYTHON} Pipeline.py'
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
}
