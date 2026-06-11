var binParam = 3;
var alpha = 5;

// State values: 18 evenly spaced points from -3 to 3 exclusive in steps of 1/3
var stateValues = map(function(i) { return -3 + (1/3) * i + 1/6; }, _.range(18));
// Actually: "18 evenly spaced points from -3 to 3 (exclusive) in steps of 1/3"
// Steps of 1/3, from -3 to 3 exclusive: -3, -8/3, -7/3, ... 
// Let me recalculate: starting at -3+1/6 would be midpoints, but "evenly spaced from -3 to 3 exclusive in steps of 1/3"
// More likely: -3 + 1/3*i for i in 0..17, but that gives -3 to 8/3... 
// 18 points, step 1/3: from -3 to -3 + 17/3 = -3 + 5.667 = 2.667 = 8/3
// Or maybe from -3+1/6 stepping by 1/3 (midpoints of bins)?
// Let's go with: stateValues = [-3 + (2i+1)/(2*3)] for i=0..17, i.e. midpoints
var stateVals = map(function(i) { return -3 + (2*i + 1) / (2 * binParam); }, _.range(18));

// Gaussian PDF
var gaussianPDF = function(x, mu, sigma) {
  return Math.exp(-0.5 * Math.pow((x - mu)/sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
};

// State probabilities for each comparison class
var superProbs = map(function(s) { return gaussianPDF(s, 0, 1); }, stateVals);
var subProbs = map(function(s) { return gaussianPDF(s, 1, 0.5); }, stateVals);

var normalize = function(arr) {
  var s = sum(arr);
  return map(function(x) { return x / s; }, arr);
};

var superProbsNorm = normalize(superProbs);
var subProbsNorm = normalize(subProbs);

// Thresholds for 'tall': state value minus 1/(2*binParam)
var tallThresholds = map(function(s) { return s - 1/(2*binParam); }, stateVals);
// Thresholds for 'short': state value plus 1/(2*binParam)
var shortThresholds = map(function(s) { return s + 1/(2*binParam); }, stateVals);

// Utterances
var utterances = ['tall', 'short', 'silence'];

// Meaning function
var meaning = function(utt, state, tallThresh, shortThresh) {
  if (utt === 'tall') { return state > tallThresh; }
  if (utt === 'short') { return state < shortThresh; }
  if (utt === 'silence') { return true; }
};

// Literal listener: given utterance and comparison class, returns distribution over states
var literalListener = mem(function(utt, compClass) {
  var probs = compClass === 'superordinate' ? superProbsNorm : subProbsNorm;
  return Infer({method: 'enumerate'}, function() {
    var stateIdx = categorical({ps: probs, vs: _.range(18)});
    var state = stateVals[stateIdx];
    var tallThreshIdx = randomInteger(18);
    var tallThresh = tallThresholds[tallThreshIdx];
    var shortThreshIdx = randomInteger(18);
    var shortThresh = shortThresholds[shortThreshIdx];
    condition(meaning(utt, state, tallThresh, shortThresh));
    return state;
  });
});

// Speaker: given state, tall threshold, short threshold, comparison class
var speaker = mem(function(state, tallThresh, shortThresh, compClass) {
  return Infer({method: 'enumerate'}, function() {
    var utt = uniformDraw(utterances);
    var ll = literalListener(utt, compClass);
    factor(alpha * ll.score(state));
    return utt;
  });
});

// Pragmatic listener hearing 'tall' about a basketball player
// The entity is a basketball player => comparison class drawn from {superordinate, subordinate}
var pragmaticListener = Infer({method: 'enumerate'}, function() {
  var compClass = uniformDraw(['superordinate', 'subordinate']);
  var probs = compClass === 'superordinate' ? superProbsNorm : subProbsNorm;
  var stateIdx = categorical({ps: probs, vs: _.range(18)});
  var state = stateVals[stateIdx];
  var tallThreshIdx = randomInteger(18);
  var tallThresh = tallThresholds[tallThreshIdx];
  var shortThreshIdx = randomInteger(18);
  var shortThresh = shortThresholds[shortThreshIdx];
  var sp = speaker(state, tallThresh, shortThresh, compClass);
  observe(sp, 'tall');
  return compClass;
});

var ANSWER = pragmaticListener;