# Background
A part of speech (POS) tagger labels each word in a sentence with its part of speech (noun, verb, etc.). For example:

The Fulton County Grand Jury said Friday an investigation of Atlanta's 
recent primary election produced `` no evidence '' that any 
irregularities took place .

=>
DET NP N ADJ N VD N DET N P NP 
ADJ N N VD `` DET N '' CNJ DET 
N VD N .

or
The/DET Fulton/NP County/N Grand/ADJ Jury/N said/VD Friday/N an/DET investigation/N of/P Atlanta's/NP
recent/ADJ primary/N election/N produced/VD ``/`` no/DET evidence/N ''/'' that/CNJ any/DET 
irregularities/N took/VD place/N ./.
So we see that "The" is a "DET" (determiner), "Fulton" an "NP" (proper noun), "County" a "N" (noun), etc. The punctuation is its own tag.

The tags that we'll use for this problem set are taken from the book Natural Language Processing with Python:

Tag	Meaning	Examples
ADJ	adjective	new, good, high, special, big, local
ADV	adverb	really, already, still, early, now
CNJ	conjunction	and, or, but, if, while, although
DET	determiner	the, a, some, most, every, no
EX	existential	there, there's
FW	foreign word	dolce, ersatz, esprit, quo, maitre
MOD	modal verb	will, can, would, may, must, should
N	noun	year, home, costs, time, education
NP	proper noun	Alison, Africa, April, Washington
NUM	number	twenty-four, fourth, 1991, 14:24
PRO	pronoun	he, their, her, its, my, I, us
P	preposition	on, of, at, with, by, into, under
TO	the word to	to
UH	interjection	ah, bang, ha, whee, hmpf, oops
V	verb	is, has, get, do, make, see, run
VD	past tense	said, took, told, made, asked
VG	present participle	making, going, playing, working
VN	past participle	given, taken, begun, sung
WH	wh determiner	who, which, when, what, where, how
A single word can have different tags in different contexts; e.g., "present" could be a verb ("I'd like to present you with this opportunity to write some great code"); an adjective ("it'll be easier if you're present at section meeting"); or a noun ("no time like the present to get started on it"). A good tagger makes use of the context to disambiguate the possibilities and determine the correct label. The tagging is in turn a core part of having a "Sudi" understand what a sentence means and how to respond.

We'll be taking the statistical approach to tagging, which means we need data. Fortunately the problem has been studied a great deal, and there exist good datasets. One prominent one is the Brown corpus (extra credit: outdo it with a Dartmouth corpus), covering a range of works from the news, fictional stories, science writing, etc. It was annotated with a more complex set of tags, which have subsequently been distilled down to the ones above.

The goal of POS tagging is to take a sequence of words and produce the corresponding sequence of tags.

# POS tagging via HMM
We will use a hidden Markov model (HMM) approach, applying the principles covered in class. Recall that in an HMM, the states are the things we don't see (hidden) and are trying to infer, and the observations are what we do see. So the observations are words in a sentence and the states are tags because the text we'll observe is not annoted with its part of speech tag (that it our program's job). We proceed through a model by moving from state to state, producing one observation per state. In this "bigram" model, each tag depends on the previous tag. Then each word depends on the tag. (For extra credit, you can go to a "trigram" model, where each tag depends on the previous two tags.) Let "#" be the tag "before" the start of the sentence. Then a model encapsulating just our example sentence is:

pos-hmm.png
The sentence follows the path: # (start) — DET ("the") — NP ("Fulton") — N ("County") — ADJ ("Grand") — N ("jury") — VD ("said") — N ("Friday") — ... — N ("place") — . (".").

An HMM is defined by its states (here parts of speech tags), transitions (here tag to tag, with weights), and observations (here tag to word, with weights). Let's consider training the pictured model from just that one sentence (i.e., knowing the corresponding words and tags, or the path). VD was used three times, tagging "said", "took", and "produced" one time each (and thus with probability 1/3). Two of its transitions were to N (once corresponding to "said Friday" and once to "took place") and one was a transition to `` (for "produced ``"). Thus the transition probabilities from VD would be 2/3 to N and 1/3 to ``. To reemphasize: probabilities are computed for the tags (i.e., the states): for each tag, the frequencies of the transitions out are divided by the total number of transitions out, and the frequencies of the words it tags are divided by the total number of words it tags. Some probabilities / log-based edge weights you can compute from this training data by counting:

tag	# trans	transitions	# obs	observations
#	1	DET = log(1/1) = 0	n/a	n/a
DET	4	N = log(3/4) = -0.12
NP = log(1/4) = -0.60	4	the = log(1/4) = -0.60
an = log(1/4) = -0.60
no = log(1/4) = -0.60
any = log(1/4) = -0.60
N	9	ADJ = log(1/9) = -0.95
VD = log(3/9) = -0.48
DET = log(1/9) = -0.95
...	9	investigation = log(1/9) = -0.95
evidence = log(1/9) = -0.95
...
VD	3	N = log(2/3) = -0.18
``= log(1/3) = -0.48	3	produced = log(1/3) = -0.48
...
NP	2	N = log(1/2) = -0.30
ADJ = log(1/2) = -0.30	2	Fulton = log(1/2) = 0.30
Atlanta's = log(1/2) = -0.30
Now, suppose we had learned the model and wanted to tag a new sentence: "The investigation produced evidence" (that's about the most sensible sentence I could come up with using that vocabulary, though of course the sentence need not be sensible — mad libs anyone?). Note that we're not going to force a "stop" state (ending with a period, question mark, or exclamation point) since the Brown corpus includes headlines that break that rule. The Viterbi algorithm starts at the # (start) state, with a score of 0, before any observation. Then to handle observation i, it propagates from each reached state at observation i-1, following each transition. The score for the next state through observation i is the sum of the score at the current state through i-1 plus the transition score from current to next plus the score of observation i in next. So:

#	observation	next state	curr state	scores: curr + transit + obs	next score
start	n/a	#	0	n/a	0
0	The	DET	#	0 + 0 + -0.60	-0.60
1	investigation	N	DET	-0.60 + -0.12 + -0.95	-1.67
NP	DET	-0.60 + -0.60 + U	-1.20 + U
2	produced	VD	N	-1.67 + -0.48 + -0.48	-2.63
DET	N	-1.68 + -0.95 + U	-2.63 + U
ADJ	N	-1.68 + -0.95 + U	-2.63 + U
NP	(-1.20 + U) + -0.30 + U	-1.50 + 2U
N	NP	(-1.20 + U) + -0.30 + U	-1.50 + 2U
...			
3	evidence	N	VD	-2.63 + -0.18 + -0.95	-3.76
DET	(-2.63 + U) + -0.12 + U	-2.75 + 2U
...		
``	VD	-2.63 + -0.48 + U	-3.11 + U
NP	DET	(-2.63 + U) + -0.60 + U	-3.23 + 2U
...			
When we're in NP, we've never actually seen the word "investigation", so what do we use as the observation probability? log(0)=-infinity, which would make it impossible, but maybe we don't want to completely rule out something that we've never seen. So let's just give it a low log probability, call it U for "unseen". It should be a negative number, perhaps worse than the observed ones but not totally out of the realm.

Notice there are some cases where we arrive at a particular next state from different previous states. E.g., we could be in N for "evidence" with the previous word tagged as wither VD or DET. In that case, we give the tag the best score over the possibilities. I showed some of the possibilities in these cases, labeling the best as "winner" and the others as "losers". In practice, the code will just keep track of the winning score and previous state.

After working through the sentence, we look at the possible states for the last observation, to see where we could end up with the best possible score. Here it's N, at (assuming U is sufficiently bad). So we know the tag for "evidence" is N. We then backtrace to where this state came from: VD. So now we know the tag for "produced" is VD. Backtracing again goes to N, so "investigation" is N. Then back to DET, so "The" is DET. Then #, so we're done.

#Testing
To assess how good a model is, we can compute how many tags it gets right and how many it gets wrong on some test sentences. (Even tougher: how many sentences does it get entirely right vs. at least one tag wrong.) It wouldn't be fair to test it on the sentences that we used to train it (though if it did poorly on those, we'd be worried).

Provided in texts.zip are sets of files, one pair with the sentences and a corresponding one with the tags to be used for training, and another pair with the sentences and tags for testing. Each line is a single sentence (or a headline), cleanly separated by whitespace into words/tokens, with punctuation also thus separated out. The example at the beginning ("The Fulton County Grand Jury...") shows the first line of the "brown-train-sentences.txt" and "brown-train-tags.txt" file (they are supposed to be on a single line each, but were split for clarity).

So use the train sentences and train tags files to generate the HMM. Then apply the HMM to each line in the test sentences file, and compare the results to the corresponding test tags line. Count the numbers of correct vs. incorrect tags.

# Implementation Notes
## Training
While we think of the model as a graph, you need not use the Graph class, and you might in fact find it easier just to keep your own Maps of transition and observation probabilities as we did with finite automata. Think first about what the mapping structure should be (from what type to what type). Recall that you can nest maps (the value for one key is itself a map). The finite automata code might be inspiring, but remember the differences (e.g., HMM observations are on states rather than edges; everything has a log-probability associated with it).
Make a pass through the training data just to count the number of times you see each transition and observation. Then go over all the states, normalizing each state's counts to probabilities (divide by the total for the state). Remember to convert to log probabilities so you can sum the scores. (FWIW, the sample solution uses natural log, not log10.)
## Viterbi tagging
While the table shows all the scores, we really need only to keep the current ones and the next ones, as shown in the pseudocode in the lecture notes. So that might simplify the representation you use.
The backtrace, on the other hand, needs to go all the way back: for observation i, for each state, what was the previous state at observation i-1 that produced the best score upon transition and observation. Note that if we index the observations by numbers, as suggested here, the representation is essentially a list of maps.
After handling the special case of the start state, start for real with observation 0 and work forward. Either consider all possible states and look back at where they could have come from, or consider all states from which to come and look forward to where they could go. In either case, be sure to find the max score (and keep track of which state gave it).
Use a constant variable for the observation of an unobserved word, and play with its value.
The backtrace starts from the state with the best score for the last observation and works back to the start state.
