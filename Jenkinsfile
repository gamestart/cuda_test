pipeline {
  agent {
    node {
      label 'GPU1'
    }
  }

  stages {
    stage('linux-gpu') {
        agent{
            docker {
                image '192.168.100.12:5000/cuda-operators/cuda-operators-x86-build:v1.0'
                registryUrl 'http://192.168.100.12:5000'
                registryCredentialsId 'cudaop-harbor'
                args '--gpus all'
                reuseNode true
                }
        }
        stages{
            stage('build'){
                steps{
                    sh '''
                    mkdir build
                    cd build
                    cmake -DCUDAOP_BUILD_TESTING=ON -DOpenCV_DIR=/opencv ..
                    make -j8
                    '''
                }
            }

            stage('test'){
                steps{
                    sh '''
                    cd build
                    ctest
                    '''
                }
            }
        }
    }
  }

  post{
      always {
        script {
          deleteDir()
        }
      }
  }
}
