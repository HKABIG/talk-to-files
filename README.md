# Talk-To-Files

## Architecture

![RAG Based PDF Retrieval System](https://github.com/HKABIG/talk-to-files/blob/main/talktofiles.png)

## Getting Started

### Project Setup with miniforge3

We will be using miniforge3 as our python based environment as Conda isn't allowed for installation on company laptops. -[1.] Download [Miniforge3](https://github.com/conda-forge/miniforge) installer. -[2.] Run the installer and follow the on-screen instructions. -[3.] During installation, make sure to check the option to add Miniforge3 to your system PATH. -[4.] Setup miniforge3 in your powershell for this you can follow [Enable Conda in powershell](https://gist.github.com/martinsotir/2bd2e16332dff71e0fa5be3ed3468a6c).

### Setting Up environment

Create a Virtual environment for you project using below command:

```
python -m venv <env-name>
.\myvenv\Scripts\activate
```

Install All the Project Dependancies using:

```
pip install -r requirements.txt
```

To use inference api's from HuggingFace Login to HuggingFace using your personal Access Token:

```
huggingface-cli login
```

Just right click to paste your Access Token.
For Detailed steps on how to get your own token follow [HuggingFace-CLI Login](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

Now you are all setup to run your first LLM powered PDF Question Answering Chatbot.

### Runing you project locally

1. Hope you have clone the project and setted up the environment, logged in using HugginFace cli and installed all the dependancies.
2. Now use the below command to run your first chainlit-app

```
chainlit run app/app.py -w
```

3. Your app is now running on port 8000.
4. provide a file-path on cmd line and start chatting with your PDF.

### Notes

1. You can play with model-kwargs parameter to increase or decrease the models level of creativity.
2. You can alos change token size and overlap size and check for interest results.
