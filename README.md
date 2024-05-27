# porori
`Porori`: Knowledge Graph Relation Extractor.

`Dotori`: Entity Recongnition and Linking.

`BonoBono`: Korean Knowledge Graph Embedding model


![image](https://github.com/Cathy-CHS/porori/assets/61447161/66cdc22b-ad12-45fb-870b-b19ebb5da5f8)


You can reproduce the results of the paper by simply running all cells in the `main_for_submit.ipynb`

You need to **use GPU**. since we will use our pretrained RE and NER model, the T4 GPU is enough. But we recommend using high RAM. so we **recommend using L4 GPU**.

Since CPU of the colab is not that fast, the KG construction time can be too long in colab. So we uploaded 

the precomputed KGs in the `precomputed_results/` folder. You can use them to reproduce the results of the paper.

To use this precomputed results, try the 'Graph analysis on Precomputed KGs' section in the main_for_submit.ipynb.


