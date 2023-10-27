# Convert Image

- Make sure python and pip was installed in your machine.
- `pip install virtualenv` Install virtualenv
- `python -m venv env` Create venv
- `source env/bin/activate` Activate venv
- `pip install -r requirements.txt` Install required package
- `mkdir -p files` Create a directory in the root of the project files.
- Put **Content** and **Style** image in `files` directory.
- `python ./app/main.py`
  - Input the content and style image with `ext`.
- After running the model the output will be stored inside files directory with name `output-<rand>.jpg`
