## neural style transfer

#### References:<br>
1. https://www.tensorflow.org/tutorials/generative/style_transfer<br>
2. https://arxiv.org/abs/1508.06576<br>
<br>

#### Requirements<br>
Python >= 3.6<br>
From terminal, run: `pip install -r requirements.txt` to install packages (including TensorFlow)<br>

#### Style your image:<br>
From terminal, run: `python main.py --content_image='{path_to_image}' --style_image='{path_to_image}' --output_dir='output/{name_of_styled_image}'`, or `python main.py` for default images. Check inside main.py for more arguments. 
