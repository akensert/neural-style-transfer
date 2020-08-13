## neural style transfer

#### References:<br>
1. https://www.tensorflow.org/tutorials/generative/style_transfer
2. https://arxiv.org/abs/1508.06576

#### Requirements<br>
* Python >= 3.6
* Python packages: from terminal run: `pip install -r requirements.txt` to install packages (including TensorFlow)

#### Style your image:<br>
From terminal run: `python main.py --content_image='{path_to_image}' --style_image='{path_to_image}' --output_dir='output/{name_of_styled_image}'`, or `python main.py` for default settings/images. Check inside `main.py` for more arguments.
