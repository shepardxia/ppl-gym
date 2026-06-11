var binParam = 3;
var alpha = 5;

// State values: 18 evenly spaced points from -3 to 3 (exclusive) in steps of 1/3
var stateValues = map(function(i) { return -3 + (1/3) * i + (1/6); }, _.range(0, 18));
// Actually let me re-read: "18 evenly spaced points from -3 to 3 (exclusive) in steps of 1/3"
// Steps of 1/3 from -3 to 3 exclusive: -3, -8/3, -7/3, ... but that's 19 points including -3
// Let me use: start = -3 + 1/(2*binParam) = -3 + 1/6, step = 1/3, 18 points
var stateVals = map(function(i) { return -3 + (1/6) + (1/3)*i; }, _.range(0, 18));

// Gaussian PDF
var gaussianPDF = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu)/sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
};

// State probabilities for superordinate: Gaussian(0, 1)
var superProbs = map(function(s) { return gaussianPDF(s, 0, 1); }, stateVals);
var superProbsNorm = (function() {
  var total = sum(superProbs);
  return map(function(p) { return p / total; }, superProbs);
})();

// State probabilities for subordinate (basketball): Gaussian(1, 0.5)
var subProbs = map(function(s) { return gaussianPDF(s, 1, 0.5); }, stateVals);
var subProbsNorm = (function() {
  var total = sum(subProbs);
  return map(function(p) { return p / total; }, subProbs);
})();

// Thresholds for 'tall': each state value minus 1/(2*binParam)
var tallThresholds = map(function(s) { return s - 1/(2*binParam); }, stateVals);
// Thresholds for 'short': each state value plus 1/(2*binParam)
var shortThresholds = map(function(s) { return s + 1/(2*binParam); }, stateVals);

var utterances = ['tall', 'short', 'silence'];

// Utterance meaning
var meaning = function(utt, state, threshold) {
  if (utt === 'tall') { return state > threshold; }
  if (utt === 'short') { return state < threshold; }
  if (utt === 'silence') { return true; }
  return true;
};

// Literal listener: given utterance and comparison class, infer state
var literalListener = mem(function(utt, cc) {
  return Infer({method: 'enumerate'}, function() {
    var probs = (cc === 'superordinate') ? superProbsNorm : subProbsNorm;
    var stateIdx = randomInteger(stateVals.length);
    var state = stateVals[stateIdx];
    factor(Math.log(probs[stateIdx]));
    
    // Sample threshold
    var threshold = (utt === 'tall') ? 
      uniformDraw(tallThresholds) :
      (utt === 'short') ? uniformDraw(shortThresholds) : 0;
    
    condition(meaning(utt, state, threshold));
    return state;
  });
});

// Speaker: given state and comparison class, choose utterance
var speaker = mem(function(state, cc) {
  return Infer({method: 'enumerate'}, function() {
    var utt = uniformDraw(utterances);
    var llScore = literalListener(utt, cc).score(state);
    factor(alpha * llScore);
    return utt;
  });
});

// Pragmatic listener: hears utterance about a basketball player
// Infers state and comparison class
// cc is drawn uniformly from {superordinate, subordinate}
// state is drawn from basketball player distribution (subordinate: Gaussian(1, 0.5))
var pragmaticListener = Infer({method: 'enumerate'}, function() {
  var cc = uniformDraw(['superordinate', 'subordinate']);
  
  // State drawn from basketball player's actual distribution (subordinate)
  var stateIdx = randomInteger(stateVals.length);
  var state = stateVals[stateIdx];
  factor(Math.log(subProbsNorm[stateIdx]));
  
  var utt = 'tall';
  var speakerDist = speaker(state, cc);
  observe(speakerDist, utt);
  
  return cc;
});

var ANSWER = pragmaticListener;