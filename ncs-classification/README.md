1. Please download the necessary files from [this](https://drive.google.com/drive/folders/19lNm7r8vYWTgz3ETGlx6_geXykJNQDqj?usp=sharing)
link and place the models, images, and graph files into the ncs-classification folder.
2. The pi_deep_learning.py file can be ran with any desktop/laptop or Pi, this is using CPU commands
  - Running SqueezeNet
    - python pi_deep_learning.py --prototxt models/squeezenet_v1.0.prototxt
	  --model models/squeezenet_v1.0.caffemodel --dim 227
	  --labels synset_words.txt --image images/image.png
  - Running GoogLeNet
    - python pi_deep_learning.py --prototxt models/bvlc_googlenet.prototxt
	  --model models/bvlc_googlenet.caffemodel --dim 224
	  --labels synset_words.txt --image images/image.png
  - Running AlexNet
    - python pi_deep_learning.py --prototxt models/bvlc_alexnet.prototxt
	  --model models/bvlc_alexnet.caffemodel --dim 227
	  --labels synset_words.txt --image images/image.png
3. The pi_ncs_deep_learning.py can be ran using the Pi with the Movidius only
  - Running SqueezeNet
    - python pi_ncs_deep_learning.py --graph graphs/squeezenetgraph
	  --dim 227 --labels synset_words.txt --image images/image.png
  - Running GoogLeNet
    - python pi_ncs_deep_learning.py --graph graphs/googlenetgraph
	  --dim 224 --labels synset_words.txt --image images/image.png
  - Running AlexNet
    - python pi_ncs_deep_learning.py --graph graphs/alexnetgraph
	  --dim 227 --labels synset_words.txt --image images/image.png
