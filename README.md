# llm-train-scripts

Short simple language model training scripts using Hugging Face's `transformers` library. These scripts are intended to be used as a starting point for training language models on custom datasets.

## Scripts

- `train_clm_eli5.py`: Train a causal language model on the ELI5 dataset
- `train_clm_eli5_loop.py`: Train a causal language model on the ELI5 dataset with a explicit training loop
- `train_clm_eli5_char.py`: Train a causal language model on the ELI5 dataset with character-level tokenization
- `train_clm_shakespeare_char.py`: Train a causal language model on the Shakespeare dataset with character-level tokenization

## Quickstart

Clone repo, install dependencies, and run script

```bash
git clone
cd llm-train-scripts
pip install -r requirements.txt
python train_clm_shakespeare_char.py
```

## Results

| Script                        | Epochs | PPL  | Loss |
| ----------------------------- | ------ | ---- | ---- |
| train_clm_shakespeare_char.py | 50     | 3.96 | 1.38 |

Sample outputs

```
First Senator:
What be the duke his for once of his honest?

VOLUMNIA:
I would have you have word in her good accountry.

All to us.

MENENIUS:
He is the enough better turn to him!

LEONTES:
As the counterfeits your come heard.

CORIOLANUS:
You say the grummat fouly y y babat he, h, hise ay dile t the thabaprow y, me! at me nthe aingouruthe I the, she, he! h, y t y t my the mbay h, bathe he habigos h cath he fow, t, he y h gust brelenth me t isthe it y mee le sthe th outhe hithe h sthen y anthoristhe! badi
```

<!-- Epoch 20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| [Duration: 00:13][Loss: , loss=1.348]
[Epoch 20] PPL: 3.96 | Loss: 1.38

His placeful some inners' moieting and stroke
The little wintend of the friends? They not cause was well brief our citizens,
About him home strew herefore that he shall propersess' sake,
As we was for the brother two to the unto his leadings.

ISABELLA:
O, GHephay oway t me y t dint it y he t sthin t y hathe me thay n Coveray he GLAt y, de main whithruthay t he he t we, th abe mitho we, he hay e min Cawhe t tholy the y the hay me caway prere I th ce habe in carin ay haron he gle min, y adinthe way t:
Ndis m

---

First Senator:
What be the duke his for once of his honest?

VOLUMNIA:
I would have you have word in her good accountry.

All to us.

MENENIUS:
He is the enough better turn to him!

LEONTES:
As the counterfeits your come heard.

CORIOLANUS:
You say the grummat fouly y y babat he, h, hise ay dile t the thabaprow y, me! at me nthe aingouruthe I the, she, he! h, y t y t my the mbay h, bathe he habigos h cath he fow, t, he y h gust brelenth me t isthe it y mee le sthe th outhe hithe h sthen y anthoristhe! badi

---

KING RICHARD II:
Suddenly I should to do hear the Duke of Norfolk,
Lord of Norfolk of Lord of York, my of Lancaster,
And to hear my name. I know no less the talk of our shepherd,
Which wear renowned and so prance of York.

KING RICHARD II:
I my lord, I cay hay sprome I g t mele, t in hay lay tharuthay t Came bre thain t that yow sthowin in ay thayowse thet y st w, w I thin thabay gristow, adamyo in thalay t n st n th thougonglay thayow, Rin boure camy, thame ay ayo hamamy w, ye w me wn Cay n my t w, y sthe

--- -->
