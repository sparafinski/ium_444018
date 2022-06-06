pipeline {
  	agent {
		dockerfile {
			 additionalBuildArgs "--build-arg KAGGLE_USERNAME=${params.KAGGLE_USERNAME} --build-arg KAGGLE_KEY=${params.KAGGLE_KEY} --build-arg CUTOFF=${params.CUTOFF} -t docker_image"
		}
	}
	parameters {
        string(
            defaultValue: 'szymonparafinski',
            description: 'Kaggle username',
            name: 'KAGGLE_USERNAME',
            trim: false
        )
        password(
            defaultValue: '',
            description: 'Kaggle token taken from kaggle.json file, as described in https://github.com/Kaggle/kaggle-api#api-credentials',
            name: 'KAGGLE_KEY'
        )
	string(
            defaultValue: '100',
            description: 'Cutoff lines',
            name: 'CUTOFF'
        )
    }
   stages {
	stage('Script'){
		steps {
            			archiveArtifacts artifacts: 'data_test.csv, data_train.csv, data_dev.csv', followSymlinks: false
		}
	}
	}
}
