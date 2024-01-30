python src/main.py --modelname bert_resnet_concat --use_image 0 --version use_text_only
python src/main.py --modelname bert_resnet_concat --use_text 0 --version use_img_only --resnet 18
python src/main.py --modelname bert_resnet_concat --use_text 0 --version use_img_only --resnet 50
python src/main.py --modelname bert_resnet_concat --version bert_resnet18_concat --resnet 18
python src/main.py --modelname bert_resnet_concat --version bert_resnet50_concat --resnet 50
python src/main.py --modelname bert_resnet_weight --version bert_resnet18_weight --resnet 18
python src/main.py --modelname bert_resnet_weight --version bert_resnet50_weight --resnet 50
python src/main.py --modelname roberta_swin_att --version roberta_swin_att 