# Language-Model

LMTrainer.py  
Thomas Giulianelli  
CSC 470 Natural Language Processing - Project 2  
10/25/16

### How to Run:

1. Ensure corpus text files are in the same directory as LMTrainer.py
2. `cd` into the directory
3. Run `python LMTrainer.py` 

The code will generate a few text files containing probability figures and output information similar to what is shown in the sample analysis section below. 
You are free to create language models with any corpus. Just save it as a text file in the same directory as LMTrainer.py and edit the main function in the code.


-----------------------------------------------------------------------------------------------

										-Analysis-
								
	Brown Corpus-								

	Most probable unigrams: [('the', 53699), ('of', 32254), ('and', 22880), ('to', 21491), ('a', 17909), ('in', 16896), ('is', 9237), ('that', 8336), ('for', 7639), ('was', 6979)]
	Most probable bigrams: [('of the', 8715), ('in the', 4833), ('to the', 2938), ('on the', 1864), ('and the', 1858), ('for the', 1580), ('to be', 1418), ('with the', 1277), ('that the', 1248), ('of a', 1247)]

	perplexityMLE = 6017.05464136
	perplexityAdd1 = 5642.36323352

	Sentences generated by language model trained on Brown Corpus:
		MLE:

		 child's performance we state religion. Its drafters discussed as rehearsed
		 married. She smiled, and Thor Hanover (Dean Hanover-Lucy Hanover), 2:30.3-:36.1;
		 vouchers certifying work on. Like his own. But why so
		 a good-will mission out conspicuously absent; the ridges fifty feet.
		 estimated. As examples is put aside from bacterial diarrhea, and
		 hides the blocks. The importance or drops in liquor suspended
		 models, in Anglo-Saxon poetry was unlikely conclusion, on extant Homeric
		 male dancers were too brave. Where sewing is ordinarily by
		 [God] is regarded the judgments to country papers were prevented
		 it', Vic theater Saturday afternoon, when interpreted as safe distance

		Add1 Smoothing:

		 with tomatoes and letters. After Thompson reported here concerned an
		 is find good team considered judgment, the genie in Katanga,
		 Papa to human organization with basic spiritual character Simon Suggs
		 she later called Bourbons. The Mayor in psychotherapy for dessert.
		 Gillis. Petitions asking the Classical World Report: "_YOUR CONGRESSMAN, SAMUEL
		 there were, Walter D& Roosevelt Memorial, a virgin earth. I
		 policy literature freed from Douglass was futile because, when they
		 years drowsed on formulating fire-resistant or trumpet vines or remedy
		 shone brightest, even non-violent means it around, either side that
		 neither chronological membership statistics which ~|a meets another nondrying oil,



	Santa Barbara Corpus-

	Most probable unigrams: [('(H)', 4325), ('the', 3746), ('I', 3072), ('--', 2383), ('you', 2198), ('to', 2181), ('and', 2147), ('a', 1948), ('of', 1641), ('that', 1575)]
	Most probable bigrams: [('you know,', 373), ('in the', 312), ("I don't", 287), ('of the', 256), ('(H) And', 247), ('I think', 201), ('(H) and', 195), ('it was', 191), ('I was', 180), ('(H) I', 177)]

	perplexityMLE = 141642834.654
	perplexityAdd1 = 134983211.975

	Sentences generated by language model trained on Santa Barbara Corpus:
		MLE:
	
		 as similar manner, but @we @are. BRAD: there's] all reached
		 experienced, (H) (SIGH)[=] BRETT: <VOX<YWN Oh no problem. SETH: Are
		 in using all around, like, old hands, I played hearts
		 patient and they've been thinking, gosh, (H) [One of tape
		 [2It's a2] [3time3]. JOANNE: (H) No=. I go, <VOX oh.
		 JILL: @@@, JEFF: [(H)] TAMMY: [3(TSK) You @know, @@@ And
		 ah, yeah I'm sorry2] [3I3] missed a spirit into play,
		 still in my husband of who we have his part
		 making positive choice. It pays <X That'll be ... objectively
		 she hires me like second time too, here. I'm bringing

		Add1 Smoothing:
	
		 PHIL: (H) U=[2m2], ARNOLD: [6Yeah sh-6], That's one six seven
		 Hanukkah (Hx) <WH @@@ PAMELA: [@] KEN: [I can] play
		 when is uh y- from .. believing in fifteen fifteen,
		 in]to compost, MARILYN: [No thank you. See we're talking like
		 FRANK: Yeah, the shower at anyone, who's a sense for
		 heading, when they, they wanted em obviously. SETH: See where
		 JAN: [(Hx)] BRETT: [3(H)3] That's yours, big gasps, like, (THROAT)
		 wh=at did X> ripped2] [3it3] off, you right before have
		 everything up of. KEVIN: @@ [3That was kind of3] board
		 (H) [5@@@5] PETE: They still] ask. LANCE: [At you]? DARRYL:
	 
-----------------------------------------------------------------------------------------------
 
### Notes:  
I couldn't get the weighted select algorithm working properly so instead I ended up just randomly choosing the words/bigrams for the sentence generation. Therefore with the current algorithm, there is no difference between the sentences generated by generateSentencesMLE() and generateSentencesAdd1().  

The test set chosen for calculating the perplexity is a sentence derived from the Brown Corpus. This is the main reason why the perplexity of the models trained on the Brown Corpus are much lower than the models trained on the Santa Barbara Corpus.  

In both cases, the perplexity of the model trained using Add1 smoothing was lower than the one trained on MLE. This was the expected result.
