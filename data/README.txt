Hindi Visual Genome 1.1 Release Description
-------------------------------------------
http://hdl.handle.net/11234/1-3267
Shantipriya Parida
Idiap Research Institute
Martigny, Switzerland

Ondrej Bojar
Charles University, Faculty of Mathematics and Physics,
Institute of Formal and Applied Linguistics (UFAL)
2020

Data
----
Hindi Visual Genome 1.1 is an updated version of Hindi Visual Genome 1.0.
The update concerns primarily the text part of Hindi Visual Genome,
fixing translation issues reported during WAT 2019 multimodal task. In
the image part, only one segment and thus one image were removed from
the dataset.

Hindi Visual Genome 1.1 serves in "WAT 2020 Multi-Modal Machine
Translation Task".

Hindi Visual Genome is a multimodal dataset consisting of text and
images suitable for English-to-Hindi multimodal machine translation task
and multimodal research. We have selected short English segments
(captions) from Visual Genome along with associated images and
automatically translated them to Hindi with manual post-editing, taking
the associated images into account.

The training set contains 29K segments. Further 1K and 1.6K segments are
provided in a development and test sets, respectively, which follow the
same (random) sampling from the original Hindi Visual Genome.

A third test set is called ``challenge test set'' consists of 1.4K
segments and it was released for WAT2019 multi-modal task. The challenge
test set was created by searching for (particularly) ambiguous English
words based on the embedding similarity and manually selecting those
where the image helps to resolve the ambiguity. The surrounding words in
the sentence however also often include sufficient cues to identify the
correct meaning of the ambiguous word.

Dataset Formats
--------------
The multimodal dataset contains both text and images.

The text parts of the dataset (train and test sets) are in simple
tab-delimited plain text files.

All the text files have seven columns as follows:

Column1 - image_id
Column2 - X
Column3 - Y
Column4 - Width
Column5 - Height
Column6 - English Text
Column7 - Hindi Text

The image part contains the full images with the corresponding image_id
as the file name. The X, Y, Width and Height columns indicate the
rectangular region in the image described by the caption.

Data Statistics
----------------
The statistics of the current release is given below.

Parallel Corpus Statistics
---------------------------

Dataset       	Segments 	English Words   	Hindi Words
-------       	---------	----------------	-------------
Train         	    28930	          143164	       145448
Dev           	      998	            4922	         4978
Test          	     1595	            7853	         7852
Challenge Test	     1400	            8186	         8639
-------       	---------	----------------	-------------
Total         	    32923	          164125	       166917

The word counts are approximate, prior to tokenization.

Citation
--------

If you use this corpus, please cite the following paper:

@article{hindi-visual-genome:2019,
  title={{Hindi Visual Genome: A Dataset for Multimodal English-to-Hindi Machine Translation}},
  author={Parida, Shantipriya and Bojar, Ond{\v{r}}ej and Dash, Satya Ranjan},
  journal={Computaci{\'o}n y Sistemas},
  volume={23},
  number={4},
  pages={1499--1505},
  year={2019}
}

