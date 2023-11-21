import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class NaturalLanguageProcessor {
    Map<String, Map<String, Integer>> transitionWords;
    Map<String, Map<String, Integer>> transitionTags;
    Map<String, Map<String, Double>> trainedWords;
    Map<String, Map<String, Double>> trainedTags;
    final double unobserved = -10;
    ArrayList<String> tags;
    ArrayList<String> sentences;
    ArrayList<String> path;

    public NaturalLanguageProcessor() throws Exception {
        tags = loadTags();
        sentences = loadSentences();
    }

    // loads tags onto the map when called
    public ArrayList<String> loadTags() throws Exception {
        tags = new ArrayList<String>();
        BufferedReader tagReader = new BufferedReader(new FileReader("inputs/texts/example-tags.txt"));

        String current = tagReader.readLine();
        while(current != null) {
            String standardizedLine = current.toLowerCase();
            String[] values = standardizedLine.split(" ");
            tags.add("#");
            for(int i = 0; i < values.length; i++) {
                tags.add(values[i]);
            }
            current = tagReader.readLine();
        }
        return tags;
    }

    // loads sentences onto the map when called
    public ArrayList<String> loadSentences() throws Exception {
        ArrayList<String> sentences = new ArrayList<String>();
        BufferedReader sentenceReader = new BufferedReader(new FileReader("inputs/texts/example-sentences.txt"));

        String current = sentenceReader.readLine();
        while(current != null) {
            String standardizedLine = current.toLowerCase();
            String[] values = standardizedLine.split(" ");
            sentences.add("#");
            for(int i = 0; i < values.length; i++) {
                sentences.add(values[i]);
            }
            current = sentenceReader.readLine();
        }
        return sentences;
    }

    // method to decode through the viterbi algorithm
    public void viterbiDecode(ArrayList<String> wordSequence) {
        // creates local variables to keep track of the backtrack, the current states, the current scores, and creates the path
        ArrayList<Map<String, String>> backTrack = new ArrayList<Map<String, String>>();
        Set<String> states = new HashSet<String>();
        HashMap<String, Double> scores = new HashMap<String, Double>();
        path = new ArrayList<String>();

        // adds the initial character to states and scores
        states.add("#");
        scores.put("#", 0.0);

        // loops though the wordSequence/sentence parameter
        for(int i = 0; i < wordSequence.size(); i++) {
            // creates variables to go back if needed and to keep track of the next scores and states
            HashMap<String, String> back = new HashMap<String, String>();
            HashMap<String, Double> nextScores = new HashMap<String, Double>();
            Set<String> nextStates = new HashSet<String>();
            // loops through each current state
            for (String currState : states) {
                // loops through each possible next state for the current tag
                for (String nextState : trainedTags.get(currState).keySet()) {
                    // creates a variable for the next score, which we will calculate, and adds the next state to the set
                    double nextScore = 0;
                    nextStates.add(nextState);
                    // if the words do not contain the key of the specific word for the next state
                    if (!trainedWords.get(nextState).containsKey(wordSequence.get(i))) {
                        // get the current state and add the trained tag value to get the next score
                        nextScore = scores.get(currState) + trainedTags.get(currState).get(nextState) + unobserved;
                    // if it does contain the key for the word, proceed
                    } else {
                        // calculate the next score by adding the current state, the next state of the current state in the tags, and the mext word value of the next state from the trained words
                        nextScore = scores.get(currState) + trainedTags.get(currState).get(nextState) + trainedWords.get(nextState).get(wordSequence.get(i));
                    }
                    // if the next scores does not have the next state, proceed
                    if (!nextScores.containsKey(nextState)) {
                        // add the calculated next state and score to the map
                        // update the back pedal accordingly
                        nextScores.put(nextState, nextScore);
                        back.put(nextState, currState);
                    }
                }
            }
            // updates the current state and score to the next state and score
            states = nextStates;
            scores = nextScores;
            // adds the back map to the list of backTrack
            backTrack.add(back);
        }

        // finds the winning tag from the trained data
        String winner = null;
        for(String word : scores.keySet()) {
            if(winner == null) winner = word;
            else if(scores.get(word) > scores.get(winner)) {
                winner = word;
            }
        }

        // back tracks through the data to find the winning tags
        String curr = winner;
        for(int i = backTrack.size() - 1; i >= 0; i--) {
            path.add(0, curr);
            curr = backTrack.get(i).get(curr);
        }
        // prints the data accordingly through the path
        for(String entry : path) {
            if(entry.equals("#")) {
                System.out.println(" ");
            }
            else if(entry.equals(".")) {
                System.out.println();
            }
            else {
                System.out.println(entry + " ");
            }
        }

    }

    // method to initialize the transition tags and words
    public void transition(ArrayList<String> trainSentence, ArrayList<String> trainTags) {
        // creates the maps
        transitionTags = new HashMap<String, Map<String, Integer>>();
        transitionWords = new HashMap<String, Map<String, Integer>>();

        // loops through the given parameter for tags
        for(int i = 0; i < trainTags.size(); i++) {
            // condition for if the next does not exist in the loop
            if ((i + 1) < trainTags.size()) {
                // if the current key exists in the map, proceed
                if (transitionTags.containsKey(trainTags.get(i))) {
                    // if the current key exists in the map within the key, proceed
                    if (transitionTags.get(trainTags.get(i)).containsKey(trainTags.get(i))) {
                        // creates a variable for the current tag value and adds it to the tag map
                        int curr = transitionTags.get(trainTags.get(i)).get(trainTags.get(i + 1)) + 1;
                        transitionTags.get(trainTags.get(i + 1)).put(trainTags.get(i), curr);
                        // if the current key within the map does not exist, add it to the map
                    } else {
                        transitionTags.get(trainTags.get(i)).put(trainTags.get(i + 1), 1);
                    }
                // if the current key does not exist, create a new entry for the map
                } else {
                    // creates a local map to add to the transition tags map
                    Map<String, Integer> tempTags = new HashMap<String, Integer>();
                    tempTags.put(trainTags.get(i + 1), 1);
                    transitionTags.put(trainTags.get(i), tempTags);
                }
            }
        }

        // does the same as the for loop above, but for the words
        for(int i = 0; i < trainSentence.size(); i++) {
            if ((i + 1) < trainSentence.size()) {
                if (transitionWords.containsKey(trainSentence.get(i))) {
                    if (transitionWords.get(trainSentence.get(i)).containsKey(trainSentence.get(i))) {
                        Integer curr = transitionWords.get(trainSentence.get(i)).get(trainSentence.get(i + 1)) + 1;
                        transitionWords.get(trainSentence.get(i + 1)).put(trainSentence.get(i), curr);
                    } else {
                        transitionWords.get(trainSentence.get(i)).put(trainSentence.get(i + 1), 1);
                    }
                } else {
                    Map<String, Integer> tempWords = new HashMap<String, Integer>();
                    tempWords.put(trainSentence.get(i + 1), 1);
                    transitionWords.put(trainSentence.get(i), tempWords);
                }
            }
        }
    }

    // method to train the processor to assign probabilities to tags
    public void train(ArrayList<String> trainSentence, ArrayList<String> trainTags) {
        // create maps for the trained tags and words
        trainedTags = new HashMap<String, Map<String, Double>>();
        trainedWords = new HashMap<String, Map<String, Double>>();

        // loops through each tag in the transition tags
        for(String tag : transitionTags.keySet()) {
            // local number to keep track of the number of transitions possible
            int num = 0;
            // loops through each string in the tag key set
            for(String curr : transitionTags.get(tag).keySet()) {
                // adds the number of transitions to the local variable
                num += transitionTags.get(tag).get(curr);
            }
            // creates a local map to track probabilities, later to add to the trained maps
            Map<String, Double> probabilities = new HashMap<String, Double>();
            // loops through each string in the key set of the tag to update it accordingly based on the number of transitions
            for(String curr : transitionTags.get(tag).keySet()) {
                // calculates the probability by dividing the tag value by the number of transitions, then taking the log
                // adds the probability and the current state to the map
                probabilities.put(curr, Math.log((double) transitionTags.get(tag).get(curr) / (double) num));
            }
            // puts the probability map and tag into the trained map
            trainedTags.put(tag, probabilities);
        }

        // does the same as the above for loop, but for words
        for(String tag : transitionWords.keySet()) {
            int num = 0;
            for(String curr : transitionWords.get(tag).keySet()) {
                num += transitionWords.get(tag).get(curr);
            }
            Map<String, Double> probabilities = new HashMap<String, Double>();
            for(String curr : transitionWords.get(tag).keySet()) {
                probabilities.put(curr, Math.log((double) transitionWords.get(tag).get(curr) / (double) num));
            }
            trainedWords.put(tag, probabilities);
        }
    }

    // method to test the decoding using scanner input
    public void consoleTest() {
        // creates a scanner and prompts the user to enter a sentence for the software to tag
        Scanner in = new Scanner(System.in);
        System.out.println("Enter a sentence to decode: ");
        // creates an array list of strings to input into the decode method
        ArrayList<String> wordSequence = new ArrayList<String>();
        // loops through the scanner input to put the words into the array list
        String current = in.nextLine();
        while(current != null) {
            String standardizedLine = current.toLowerCase();
            String[] values = standardizedLine.split(" ");
            for(int i = 0; i < values.length; i++) {
                wordSequence.add(values[i]);
            }
            current = in.nextLine();
        }
        // decodes the sentences
        viterbiDecode(wordSequence);

    }

    // method to test the decoding using file input
    public void fileTest(String pathname) throws Exception {
        // creates a reader to get the information from the file
        BufferedReader rdr = new BufferedReader(new FileReader(pathname));
        // creates an array list of strings to input into the decode method
        ArrayList<String> wordSequence = new ArrayList<String>();
        // loops through the file data to put the words into the array list
        String current = rdr.readLine();
        while(current != null) {
            String standardizedLine = current.toLowerCase();
            String[] values = standardizedLine.split(" ");
            for(int i = 0; i < values.length; i++) {
                wordSequence.add(values[i]);
            }
            current = rdr.readLine();
        }
        // decodes the sentences
        viterbiDecode(wordSequence);
    }

    public void performanceEval() throws Exception {
        // ints to track the number of correct and incorrect tags
        int match = 0;
        int noMatch = 0;
        BufferedReader rdr = new BufferedReader(new FileReader("inputs/texts/example-tags.txt"));
        // creates an array list of strings to input into the decode method
        ArrayList<String> wordSequence = new ArrayList<String>();
        // loops through the file data to put the words into the array list
        String current = rdr.readLine();
        while(current != null) {
            String standardizedLine = current.toLowerCase();
            String[] values = standardizedLine.split(" ");
            for(int i = 0; i < values.length; i++) {
                wordSequence.add(values[i]);
            }
            current = rdr.readLine();
        }

        // loops through the path to check if the tags match the correct tag
        for(int i = 0; i < path.size(); i++) {
            if((i + 1) < path.size()) {
                if(wordSequence.get(i).equals(wordSequence.get(i+1))) {
                    match++;
                }
                else {
                    noMatch++;
                }
            }
        }

        System.out.println("There were " + match + " matches and " + noMatch + " non-matches.");
    }

    public static void main(String[] args) throws Exception {
        // creates the sudi assistant
        NaturalLanguageProcessor sudi = new NaturalLanguageProcessor();
        // instantiates the tags and sentences for the software to decode
        sudi.tags = sudi.loadTags();
        sudi.sentences = sudi.loadSentences();

        // trains sudi using the loaded sentences and tags
        sudi.transition(sudi.sentences, sudi.tags);
        sudi.train(sudi.sentences, sudi.tags);
        sudi.viterbiDecode(sudi.sentences);
        sudi.performanceEval();

    }
}
