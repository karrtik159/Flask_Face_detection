# Flask_Face_detection

# 1. Preparation

## 1.1 Quick Start

Just Complete the five steps to launch the Flask App.

1. Download the project locally by clonning or in a zip format.

2. Donwload the Emotion prediction model .h5 file and copy its path.
    
    2.1 go to the directory
    ```
    cd Flask_Face_detection
    ```
    
    2.2 go to the directory
    ```
    cd "Face and Emotion prediction"
    ```

3. Create a new virtual environment or proceed it in your current environment by running 

    3.1 For Creating virtual environment
        
        1. First install virtualenv
        
        --> pip install virtualenv
        
        2. Create a virtual environment.
        
        --> virtualenv my_env
        
        3. Now activate the environment by going to ./my_env/Scripts
 
        --> activate
    
    3.2 For installing dependicies
        
        Now after returning to the original directory which is Face and Emotion prediction.
        Run the below command
        
        --> pip install -r requirements.txt
        
4. After installing dependencies and replacing the path in app.py file for loading the model file.

5. For running the webpage, execute the following.

```
python app.py

```
Samples:
1. Home Page

![image](https://github.com/karrtik159/Flask_Face_detection/assets/65113086/427a8923-2501-4330-b5ea-81117d214701)

2. Prediction Page.

![image](https://github.com/karrtik159/Flask_Face_detection/assets/65113086/06aa59cd-9f90-46af-a22e-afe240b14296)


3. Checking working of Camera Feed

![image](https://github.com/karrtik159/Flask_Face_detection/assets/65113086/a7fc1e3a-6bae-47bc-8058-a719f7e50421)


Note:
My Model is customised for returning values like confidence score for Image which was trained on EMOTIC dataset with custom modifications and now that models are used in this.

