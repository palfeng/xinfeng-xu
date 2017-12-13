# xinfeng-xu

This is the code for paper: Hashing for Fast and Efficient Node Embedding, which is using hash functions to speed up the embedding algorithm. In our experiement, we have shown that the hash embedding methods gives similar accuracy but costs much shorter time. 

Project done by Xinfeng Xu, Virginia Tech. Supervised by B. Aditya Prakash, Virginia Tech.


All code are in src directory. Code running on Python 3.5.2 :: Anaconda custom (x86_64) and Keras 1.2.2.

Example of running:

1. python HashingNetEmbed.py

2. Then the program will ask you to input the dataset path:  “Graph color classify: Please input the filename of dataset:”, you can simply key in “./data/polblogs”, then click enter. (Careful about: don’t add space after the path, or it will show error)

3. The program will give you results on classifications accuracy, lose, and running time
