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

![image](https://user-images.githubusercontent.com/65113086/234911465-24dacaea-2f10-4bdf-8723-b71610a7fabe.png)

2. To analyze by uploading image, Use Watch button.

![image](https://user-images.githubusercontent.com/65113086/234912106-1236f4fa-1e1b-48a6-9ac4-efeae54b6c32.png)

3. To analyze with web camera, Use Analyze button.

![image](https://user-images.githubusercontent.com/65113086/234914457-c1b12fe8-77ec-48cb-83ff-b968452eee68.png)

Note:
My Model is customised for returning values like confidence score for Image which was trained on EMOTIC dataset with custom modifications and now that models are used in this.

