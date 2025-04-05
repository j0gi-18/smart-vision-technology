# smart-vision-technology

## Implements YOLO v11 trained on custom dataset for identifiying grossery items and identifying shelf life of preishable items like fruits and vegetables. Includes quality assurance measures like verifying ordered items are packed and sent for delivery carefully, checks for damaged packaging of the item , etc. Automates the process of quality assurance by using YOLO v11.


1. Trained Yolo v11 medium model on custom built dataset containing 3000+ images and attained an mean average precision score of 85%
2. Model draws bounding boxes on every recognised grocery item
3. The dataset contain 3000+ images of products ranging from staples to packaged food. 
4. Products form wide variety categorised in 80 classes from iconic brands like Amul, Britannia, Lays, etc
5. Implemented quality testing assuring item moves from the desired region and count any object moving in or out of the determined region.
6. Used easyOCR to extract all available textual information from the present product image and print at the output terminal
7. Also trained a model to predict the freshness of fruits which was trained on Fruitectives dataset contains 13000+ images of 8 types of fruits categorised into four sub categories i.e ripe, over-ripe, unripe, rotten. 
8. The fruit freshness prediction model achieved a mean average precision score of 78%
