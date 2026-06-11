var binParam = 3;
var alpha = 5;

// State values: 18 evenly spaced points from -3 to 3 (exclusive) in steps of 1/3
var states = _.range(18).map(function(i) { return -3 + (i + 0.5) * (1/3); });
// Actually, let me re-read: "18 evenly spaced points from -3 to 3 (exclusive) in steps of 1/3"
// Steps of 1/3 from -3 to 3 exclusive: -3, -3+1/3, -3+2/3, ... but that's 18 points if we go -3 to just under 3
// Let me use: -3 + i*(1/3) for i = 0..17 (18 points)
var stateValues = _.range(18).map(function(i) { return -3 + i * (1/3); });

// Gaussian PDF
var gaussianPDF = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu)/sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
};

// State probabilities for superordinate (mu=0, sigma=1)
var superProbsRaw = map(function(s) { return gaussianPDF(s, 0, 1); }, stateValues);
var superProbsSum = sum(superProbsRaw);
var superProbs = map(function(p) { return p / superProbsSum; }, superProbsRaw);

// State probabilities for subordinate basketball player (mu=1, sigma=0.5)
var subProbsRaw = map(function(s) { return gaussianPDF(s, 1, 0.5); }, stateValues);
var subProbsSum = sum(subProbsRaw);
var subProbs = map(function(p) { return p / subProbsSum; }, subProbsRaw);

// Thresholds for 'tall': each state value minus 1/(2*binParam)
var tallThresholds = map(function(s) { return s - 1/(2*binParam); }, stateValues);
// Thresholds for 'short': each state value plus 1/(2*binParam)
var shortThresholds = map(function(s) { return s + 1/(2*binParam); }, stateValues);

var utterances = ['tall', 'short', 'silence'];

// Utterance meaning given state and threshold
var meaning = function(utt, state, threshold) {
  if (utt === 'tall') { return state > threshold; }
  if (utt === 'short') { return state < threshold; }
  if (utt === 'silence') { return true; }
};

// Literal listener: given utterance and comparison class,
// returns distribution over states
var literalListener = mem(function(utt, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var stateIndex = randomInteger(18);
    var state = stateValues[stateIndex];
    var stateProb = compClass === 'superordinate' ? superProbs[stateIndex] : subProbs[stateIndex];
    factor(Math.log(stateProb));
    
    // Sample threshold uniformly
    var thresholdIndex = randomInteger(18);
    var threshold = utt === 'tall' ? tallThresholds[thresholdIndex] :
                    utt === 'short' ? shortThresholds[thresholdIndex] : 0;
    
    condition(meaning(utt, state, threshold));
    
    return state;
  });
});

// Speaker: given state and comparison class, chooses utterance
var speaker = mem(function(state, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var utt = uniformDraw(utterances);
    var ll = literalListener(utt, compClass);
    factor(alpha * ll.score(state));
    return utt;
  });
});

// Pragmatic listener: hears utterance about a basketball player
// infers state and comparison class
var pragmaticListener = function(utt) {
  return Infer({method: 'enumerate'}, function() {
    // Entity is a basketball player, so state drawn from subordinate distribution
    var stateIndex = randomInteger(18);
    var state = stateValues[stateIndex];
    factor(Math.log(subProbs[stateIndex]));
    
    // Comparison class drawn uniformly
    var compClass = uniformDraw(['superordinate', 'subordinate']);
    
    var sp = speaker(state, compClass);
    observe(sp, utt);
    
    return compClass;
  });
};

var ANSWER = pragmaticListener('tall');